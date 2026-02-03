# Copyright 2025 RLLM Team
# Shared Environment Manager for efficient environment reuse
#
# This module provides a singleton manager that maintains shared environment instances
# to avoid repeated initialization overhead (e.g., loading Webshop product data).

import logging
import threading
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class SharedEnvironmentManager:
    """
    Singleton manager for shared environment instances.
    
    This manager maintains a pool of initialized environment instances that can be
    reused across multiple CompositeEnvironment instances, avoiding the overhead
    of repeated initialization (e.g., loading Webshop product data for each task).
    
    Usage:
        # Get the singleton instance
        manager = SharedEnvironmentManager.get_instance()
        
        # Pre-initialize environments (optional, for eager loading)
        manager.initialize_env("webshop", WebshopEnvironment, webshop_args)
        
        # Get or create an environment
        env = manager.get_or_create_env("webshop", WebshopEnvironment, webshop_args)
        
        # The environment is now shared and won't be re-initialized
    
    Thread Safety:
        This manager uses locks to ensure thread-safe access to shared environments.
        However, the environments themselves may not be thread-safe (e.g., Webshop).
        For non-thread-safe environments, use n_parallel_agents=1.
    """
    
    _instance: Optional["SharedEnvironmentManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._env_cache: Dict[str, Any] = {}
        self._env_configs: Dict[str, Dict[str, Any]] = {}
        self._env_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._initialized = True
        
        logger.info("[SharedEnvManager] Initialized singleton instance.")
    
    @classmethod
    def get_instance(cls) -> "SharedEnvironmentManager":
        """Get the singleton instance of SharedEnvironmentManager."""
        return cls()
    
    def get_or_create_env(
        self,
        env_type: str,
        env_class: Type,
        env_args: Dict[str, Any],
        force_new: bool = False
    ) -> Any:
        """
        Get an existing environment or create a new one.
        
        Args:
            env_type: Type identifier for the environment (e.g., "webshop", "math")
            env_class: The environment class to instantiate
            env_args: Arguments to pass to the environment constructor
            force_new: If True, always create a new instance (ignores cache)
            
        Returns:
            The environment instance (shared or new)
        """
        if force_new:
            return self._create_env(env_type, env_class, env_args)
        
        # Ensure we have a lock for this env type
        with self._global_lock:
            if env_type not in self._env_locks:
                self._env_locks[env_type] = threading.Lock()
        
        # Check cache with lock
        with self._env_locks[env_type]:
            if env_type in self._env_cache:
                logger.debug(f"[SharedEnvManager] Reusing cached environment: {env_type}")
                return self._env_cache[env_type]
            
            # Create new environment
            env = self._create_env(env_type, env_class, env_args)
            self._env_cache[env_type] = env
            self._env_configs[env_type] = env_args.copy()
            
            logger.info(f"[SharedEnvManager] Created and cached environment: {env_type}")
            return env
    
    def _create_env(self, env_type: str, env_class: Type, env_args: Dict[str, Any]) -> Any:
        """Create a new environment instance."""
        # Handle reward_fn specially
        if "reward_fn" in env_args:
            reward_fn = env_args["reward_fn"]
            args_copy = env_args.copy()
            args_copy.pop("reward_fn", None)
            return env_class(reward_fn=reward_fn, **args_copy)
        else:
            if hasattr(env_class, "from_dict"):
                return env_class.from_dict(env_args)
            else:
                return env_class(**env_args)
    
    def initialize_env(
        self,
        env_type: str,
        env_class: Type,
        env_args: Dict[str, Any]
    ) -> Any:
        """
        Pre-initialize an environment (eager loading).
        
        This is useful for environments with heavy initialization (like Webshop)
        to ensure they are ready before processing tasks.
        
        Args:
            env_type: Type identifier for the environment
            env_class: The environment class to instantiate
            env_args: Arguments to pass to the environment constructor
            
        Returns:
            The initialized environment instance
        """
        logger.info(f"[SharedEnvManager] Pre-initializing environment: {env_type}")
        return self.get_or_create_env(env_type, env_class, env_args)
    
    def has_env(self, env_type: str) -> bool:
        """Check if an environment type is already cached."""
        return env_type in self._env_cache
    
    def get_cached_env(self, env_type: str) -> Optional[Any]:
        """Get a cached environment without creating a new one."""
        return self._env_cache.get(env_type)
    
    def close_env(self, env_type: str):
        """Close and remove a specific environment from cache."""
        with self._global_lock:
            if env_type in self._env_cache:
                env = self._env_cache.pop(env_type)
                if hasattr(env, 'close'):
                    try:
                        env.close()
                    except Exception as e:
                        logger.warning(f"[SharedEnvManager] Error closing {env_type}: {e}")
                self._env_configs.pop(env_type, None)
                logger.info(f"[SharedEnvManager] Closed environment: {env_type}")
    
    def close_all(self):
        """Close all cached environments."""
        with self._global_lock:
            for env_type in list(self._env_cache.keys()):
                env = self._env_cache.pop(env_type)
                if hasattr(env, 'close'):
                    try:
                        env.close()
                    except Exception as e:
                        logger.warning(f"[SharedEnvManager] Error closing {env_type}: {e}")
            self._env_configs.clear()
            logger.info("[SharedEnvManager] Closed all environments.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cached environments."""
        return {
            "cached_env_types": list(self._env_cache.keys()),
            "num_cached": len(self._env_cache),
        }


# Convenience function for getting the singleton
def get_shared_env_manager() -> SharedEnvironmentManager:
    """Get the singleton SharedEnvironmentManager instance."""
    return SharedEnvironmentManager.get_instance()
