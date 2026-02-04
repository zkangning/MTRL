from gym.envs.registration import register

from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
from web_agent_site.envs.web_agent_text_env import (
    WebAgentTextEnv,
    SharedWebshopData,
    get_shared_webshop_data,
)

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)

__all__ = [
    'WebAgentSiteEnv',
    'WebAgentTextEnv',
    'SharedWebshopData',
    'get_shared_webshop_data',
]