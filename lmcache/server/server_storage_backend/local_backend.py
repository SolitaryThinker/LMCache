import os
from typing import List, Optional, Set, OrderedDict
from collections import OrderedDict

from lmcache.logging import init_logger
from lmcache.server.server_storage_backend.abstract_backend import \
    LMSBackendInterface
from lmcache.storage_backend.evictor import LRUEvictor
from lmcache.storage_backend.evictor.base_evictor import PutStatus
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class LMSLocalBackend(LMSBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu 
    memory.
    """

    def __init__(self, ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current 
            configuration
        """
        super().__init__()

        self.dict: OrderedDict[CacheEngineKey, bytes] = OrderedDict()
        
        self.evictor = LRUEvictor()

    def list_keys(self) -> List[str]:

        return list(self.dict.keys())

    def contains(
        self,
        key: str,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict
    
    def remove(
        self, 
        key: CacheEngineKey,
    ) -> None:
        """
        Remove the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        """
        self.dict.pop(key)

    def put(
        self,
        key: str,
        kv_chunk_bytes: bytes,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested 
            tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if not blocking:
            logger.warn("Non-blocking is not implemented for local backend")
        
        # Obtain keys to evict
        evict_keys, put_status = self.evictor.update_on_put(self.dict, kv_chunk_local)
        
        # Abort put if cache too big
        if put_status == PutStatus.ILLEGAL:
            return
        
        # Evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)
        
        # Store new chunk
        self.dict[key] = kv_chunk_bytes

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: str,
    ) -> Optional[bytes]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        kv_chunk =self.dict.get(key, None)
        
        # Update cache recency
        if kv_chunk is not None:
            self.evictor.update_on_get(key, self.dict)

        return kv_chunk

    def close(self):
        pass


# TODO(Jiayi): need to optimize disk loading
# current impl. with "naive open read/write" might not be efficient
# (better than torch.load)
class LMSLocalDiskBackend(LMSBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local disk.
    """

    def __init__(
        self,
        path: str,
    ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
            configuration
        """
        super().__init__()

        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.filenames: Set[str] = set()

    def list_keys(self) -> List[str]:

        return list(self.filenames)

    def contains(
        self,
        key: str,
    ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.filenames

    def _key_to_path(
        self,
        key: str,
    ) -> str:
        """
        Convert key to path_name

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            returns the path name
        """
        return self.path + key.replace("/", "-") + ".bin"

    def put(
        self,
        key: str,
        kv_chunk_bytes: bytes,
        blocking: bool = True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested 
            tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if not blocking:
            logger.warn("Non-blocking is not implemented for local backend")
        self.filenames.add(key)
        logger.info(f"Saving cache to {self._key_to_path(key)}")
        # torch.save(kv_chunk_bytes, self._key_to_path(key))
        with open(self._key_to_path(key), "wb") as binary_file:
            binary_file.write(kv_chunk_bytes)

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: str,
    ) -> Optional[bytes]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output:
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        if key not in self.filenames:
            return None

        with open(self._key_to_path(key), "rb") as binary_file:
            return binary_file.read()

        # return torch.load(self._key_to_path(key))

    def close(self):
        pass
