import gym
import json
import random
import string
import time
import torch
import threading
import copy

import numpy as np

from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import defaultdict
from flask import Flask
from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    get_top_n_product_from_keywords,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    ACTION_TO_TEMPLATE,
    END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH,
)
from web_agent_site.engine.goal import get_reward, get_goals
from web_agent_site.utils import (
    DEFAULT_FILE_PATH,
    DEFAULT_ATTR_PATH,
    FEAT_CONV,
    FEAT_IDS,
    random_idx
)

app = Flask(__name__)


# =============================================================================
# 全局共享数据缓存（用于优化多实例环境的初始化性能）
# =============================================================================
# 这些数据是只读的，可以在多个 SimServer 实例之间安全共享：
# - all_products: 产品信息列表
# - product_item_dict: ASIN -> 产品信息的映射
# - product_prices: 产品价格信息
# - search_engine: Lucene 搜索引擎
# - goals: 任务目标列表
#
# 每个 SimServer 实例只需要维护自己的 user_sessions（可变状态）
# =============================================================================

class SharedWebshopData:
    """
    进程级别的共享数据缓存。
    
    所有 SimServer 实例共享这些只读数据，避免重复加载：
    - 产品数据 (~1-2s 加载时间)
    - 搜索引擎 (~0.5s 加载时间)
    - 目标列表 (~1-3s 生成时间)
    
    线程安全说明：
    - 初始化使用锁保护，确保只加载一次
    - 加载完成后数据只读，多线程访问安全
    """
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._data_cache = {}
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_or_load_data(
        self,
        file_path: str,
        attr_path: str,
        num_products: int,
        human_goals: bool,
        seed: int,
    ):
        """
        获取或加载共享数据。
        
        使用配置参数生成缓存键，相同配置只加载一次。
        
        Args:
            file_path: 产品数据文件路径
            attr_path: 属性数据文件路径
            num_products: 产品数量限制
            human_goals: 是否使用人工标注的目标
            seed: 随机种子（用于目标列表排序）
        
        Returns:
            tuple: (all_products, product_item_dict, product_prices, search_engine, goals, weights, cum_weights)
        """
        # 生成缓存键
        cache_key = (file_path, attr_path, num_products, human_goals, seed)
        
        if cache_key in self._data_cache:
            # Cache hit - no need to print every time (too verbose in training)
            return self._data_cache[cache_key]
        
        with self._lock:
            # 双重检查锁定
            if cache_key in self._data_cache:
                return self._data_cache[cache_key]
            
            print(f"[SharedWebshopData] Loading shared data: num_products={num_products}, human_goals={human_goals}")
            load_start = time.time()
            
            # 1. 加载产品数据
            all_products, product_item_dict, product_prices, _ = load_products(
                filepath=file_path,
                attrpath=attr_path,
                num_products=num_products,
                human_goals=human_goals
            )
            
            # 2. 初始化搜索引擎
            search_engine = init_search_engine(num_products=num_products)
            
            # 3. 生成目标列表
            goals = get_goals(all_products, product_prices, human_goals)
            
            # 4. 使用固定种子排序目标（确保可重复性）
            random.seed(seed)
            goals = goals.copy()  # 创建副本以便排序
            random.shuffle(goals)
            
            # 5. 预计算权重
            weights = [goal['weight'] for goal in goals]
            cum_weights = [0] + np.cumsum(weights).tolist()
            
            load_time = time.time() - load_start
            print(f"[SharedWebshopData] Loaded {len(all_products)} products, {len(goals)} goals in {load_time:.1f}s")
            
            # 缓存数据
            self._data_cache[cache_key] = (
                all_products,
                product_item_dict,
                product_prices,
                search_engine,
                goals,
                weights,
                cum_weights
            )
            
            return self._data_cache[cache_key]
    
    def clear_cache(self):
        """清除所有缓存数据（用于测试或内存回收）"""
        with self._lock:
            self._data_cache.clear()
            print("[SharedWebshopData] Cache cleared")


def get_shared_webshop_data():
    """获取共享数据管理器的便捷函数"""
    return SharedWebshopData.get_instance()
class WebAgentTextEnv(gym.Env):
    """Gym environment for Text mode of WebShop environment"""
    def __init__(
            self,
            observation_mode='html',
            file_path=DEFAULT_FILE_PATH,
            attr_path=DEFAULT_ATTR_PATH,
            server=None,
            **kwargs
        ):
        """
        Constructor for text environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        super(WebAgentTextEnv, self).__init__()
        
        # Define action_space and observation_space for gym compatibility
        # Since this is a text-based environment, we use a simple Text space
        # or a placeholder that satisfies gym's requirements
        try:
            # gym >= 0.25.0 has Text space
            self.action_space = gym.spaces.Text(max_length=1024)
            self.observation_space = gym.spaces.Text(max_length=65536)
        except AttributeError:
            # Fallback for older gym versions - use Discrete as placeholder
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Discrete(1)
        
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        self._seed = kwargs.get('seed', 42)

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        self.file_path = file_path
        self.attr_path = attr_path

        self.base_url = 'http://127.0.0.1:3000'
        self.server = SimServer(
            self._seed,
            self.base_url,
            self.file_path,
            self.attr_path,
            self.kwargs.get('filter_goals'),
            self.kwargs.get('limit_goals', -1),
            self.kwargs.get('num_products'),
            self.kwargs.get('human_goals'),
            self.kwargs.get('show_attrs', False),
        ) if server is None else server
        self.browser = SimBrowser(self.server)

        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')
        if self.kwargs.get('get_image', 0):
            self.feats = torch.load(FEAT_CONV)
            self.ids = torch.load(FEAT_IDS)
            self.ids = {url: idx for idx, url in enumerate(self.ids)}
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)
        self.reset()

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        available_actions = self.get_available_actions()

        # Determine action type (click, search) and argument
        action_name, action_arg = parse_action(action)
        
        # Normalize action_arg to lowercase for matching
        if action_arg is not None:
            action_arg = action_arg.strip().lower()
        
        # Normalize action_name to lowercase
        if action_name is not None:
            action_name = action_name.strip().lower()
        
        # Debug logging (can be disabled in production)
        # print(f"[WebShop DEBUG] action={action}, parsed: name={action_name}, arg={action_arg}")
        # print(f"[WebShop DEBUG] available clickables: {list(self.text_to_clickable.keys())[:10]}...")
        
        if (action_name == 'search' and
            action_arg is not None and
            action_arg != ''):
            status = self.browser.search(action_arg)
        elif action_name == 'click' and action_arg is not None:
            # Try exact match first
            if action_arg in self.text_to_clickable:
                status = self.browser.click(action_arg, self.text_to_clickable)
            else:
                # Try case-insensitive matching for clickables
                matched_key = None
                for key in self.text_to_clickable.keys():
                    if key.lower() == action_arg.lower():
                        matched_key = key
                        break
                
                if matched_key is not None:
                    status = self.browser.click(matched_key, self.text_to_clickable)
                else:
                    # No match found
                    # print(f"[WebShop DEBUG] No match for click[{action_arg}] in available actions")
                    status = dict(reward=0, done=False)
        else:
            status = dict(reward=0, done=False)

        # Update observation, state with the new action
        ob = self.observation
        text_list = [ob]
        self.prev_actions.append(action)
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                text_list.append(self.prev_actions[-i])
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                text_list.append(self.prev_obs[-i])
        state = ' [SEP] '.join(text_list[::-1])
        self.prev_obs.append(ob)
        
        done = status['done']
        # NOTE: Removed auto-reset on done!
        # The original code would call self.reset() here, which causes session pollution
        # in multi-task training scenarios. The reset should be handled explicitly by
        # the upper layer (WebshopEnvironment.reset()) to ensure proper session isolation.
        #
        # Original problematic code:
        # if done:
        #     self.reset()
        
        # Return info as dict for gym >= 0.25.0 compatibility
        info = {
            'reward': status['reward'],
            'done': done,
            'action': action,
            'action_name': action_name,
            'action_arg': action_arg,
        }
        return state, status['reward'], done, info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id='search_input')
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_='btn')
        product_links = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {}
        
        # Add buttons (e.g., "Back to Search", "Next >", "< Prev", "Buy Now")
        for b in buttons:
            text = b.get_text().strip().lower()
            if text:
                self.text_to_clickable[text] = b
        
        # Add product links (ASINs)
        for link in product_links:
            # Product links contain ASIN codes
            text = link.get_text().strip().lower()
            if text:
                self.text_to_clickable[text] = link
        
        # Add buying options (size, color, etc.)
        for opt in buying_options:
            opt_value = opt.get('value')
            if opt_value:
                # Store both original and lowercase versions for matching
                self.text_to_clickable[opt_value.lower()] = opt
        
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )
    
    def get_image(self):
        """Scrape image from page HTML and return as a list of pixel values"""
        html_obj = self._parse_html(self.browser.page_source)
        image_url = html_obj.find(id='product-image')
        if image_url is not None:
            image_url = image_url['src']
            if image_url in self.ids:
                image_idx = self.ids[image_url]
                image = self.feats[image_idx]
                return image
        return torch.zeros(512)

    def get_instruction_text(self):
        """Get corresponding instruction text for current environment session"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj
    
    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == 'text_rich':
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == 'url':
            return self.state['url']
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )
    
    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )
    
    def convert_html_to_text(self, html, simple=False):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            for t in visible_texts:
                if t == '\n': continue
                if t.parent.name == 'button':  # button
                    processed_t = f'[button] {t} [button_]'
                elif t.parent.name == 'label':  # options
                    if f'"{t}"' in self.state['url']:
                        processed_t = f'  [clicked button] {t} [clicked button_]'
                        observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'  [button] {t} [button_]'
                elif t.parent.get('class') == ["product-link"]: # product asins
                    if f'{t}' in self.server.user_sessions[self.session]['asins']:
                        processed_t = f'\n[clicked button] {t} [clicked button_]'
                    else:
                        processed_t = f'\n[button] {t} [button_]'
                else: # regular, unclickable text
                    processed_t =  str(t)
                observation += processed_t + '\n'
            return observation
    
    def reset(self, session=None, instruction_text=None, **kwargs):
        """Create a new session and reset environment variables
        
        Note: **kwargs added for gym >= 0.25.0 compatibility (seed, options parameters)
        """
        session_int = None
        if session is not None:
            self.session = str(session)
            if isinstance(session, int):
                session_int = session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=10))
        if self.session_prefix is not None:
            self.session = self.session_prefix + self.session

        init_url = f'{self.base_url}/{self.session}'
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        # Initialize text_to_clickable as empty dict instead of None
        # This prevents TypeError in get_available_actions() when called before step()
        self.text_to_clickable = {}
        self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
        obs = self.observation
        self.prev_obs = [obs]
        self.prev_actions = []
        
        # Return info as dict for gym >= 0.25.0 compatibility
        info = {
            'session': self.session,
            'instruction_text': self.instruction_text,
        }
        return obs, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


class SimServer:
    """
    Lightweight simulator of WebShop Flask application for generating HTML observations.
    
    性能优化说明：
    使用 SharedWebshopData 单例共享只读数据（产品、搜索引擎、目标列表），
    避免每个实例都重复加载。每个实例只维护自己的可变状态（user_sessions）。
    
    这使得创建多个 SimServer 实例的时间从 ~3-5s 降低到 <0.1s（首次除外）。
    """
    def __init__(
        self,
        seed,
        base_url,
        file_path,
        attr_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,
        human_goals=0,
        show_attrs=False,
    ):
        """
        Constructor for simulated server serving WebShop application
        
        Arguments:
        filter_goals (`func`) -- Select specific goal(s) for consideration based on criteria of custom function
        limit_goals (`int`) -- Limit to number of goals available
        num_products (`int`) -- Number of products to search across
        human_goals (`bool`) -- If true, load human goals; otherwise, load synthetic goals
        """
        self.base_url = base_url
        self.show_attrs = show_attrs
        
        # ========== 使用共享数据缓存 ==========
        # 产品数据、搜索引擎、目标列表都是只读的，可以安全共享
        shared_data = get_shared_webshop_data()
        (
            self.all_products,
            self.product_item_dict,
            self.product_prices,
            self.search_engine,
            shared_goals,
            shared_weights,
            shared_cum_weights
        ) = shared_data.get_or_load_data(
            file_path=file_path,
            attr_path=attr_path,
            num_products=num_products,
            human_goals=human_goals,
            seed=seed
        )
        
        # 目标列表可能需要过滤或限制，所以创建副本
        self.goals = shared_goals
        self.weights = shared_weights
        self.cum_weights = shared_cum_weights

        # Apply `filter_goals` parameter if exists to select specific goal(s)
        if filter_goals is not None:
            self.goals = [
                goal for (i, goal) in enumerate(self.goals)
                if filter_goals(i, goal)
            ]
            # 重新计算权重
            self.weights = [goal['weight'] for goal in self.goals]
            self.cum_weights = [0] + np.cumsum(self.weights).tolist()
        
        # Imposes `limit` on goals via random selection
        if limit_goals != -1 and limit_goals < len(self.goals):
            # 使用独立的随机状态避免影响其他组件
            rng = random.Random(seed)
            idxs = []
            while len(idxs) < limit_goals:
                idx = random_idx(self.cum_weights)
                if idx not in idxs:
                    idxs.append(idx)
            self.goals = [self.goals[i] for i in idxs]
            # 重新计算权重
            self.weights = [goal['weight'] for goal in self.goals]
            self.cum_weights = [0] + np.cumsum(self.weights).tolist()
        
        # Only print on first initialization (tracked by SharedWebshopData cache)
        # Suppress verbose logging during training

        # ========== 实例私有的可变状态 ==========
        # 每个 SimServer 实例有自己独立的 user_sessions
        self.user_sessions = dict()
        self.search_time = 0
        self.render_time = 0
        self.sample_time = 0
        self.assigned_instruction_text = None  # TODO: very hacky, should remove
        
    @app.route('/', methods=['GET', 'POST'])
    def index(self, session_id, **kwargs):
        """Redirect to the search page with the given session ID"""
        html = map_action_to_html(
            'start',
            session_id=session_id,
            instruction_text=kwargs['instruction_text'],
        )
        url = f'{self.base_url}/{session_id}'
        return html, url
    
    @app.route('/', methods=['GET', 'POST'])
    def search_results(self, session_id, **kwargs):
        """Initialize session and return the search results page"""
        session = self.user_sessions[session_id]
        keywords = kwargs['keywords']  # TODO: why is this using kwargs? why not session?
        assert isinstance(keywords, list)
        page = 1 if 'page' not in kwargs else kwargs['page']
        session["page"] = page
        session["keywords"] = keywords
        session["actions"]["search"] += 1
        session["asin"] = None
        session["options"] = {}

        # Perform search on keywords from items and record amount of time it takes
        old_time = time.time()
        top_n_products = get_top_n_product_from_keywords(
            keywords,
            self.search_engine,
            self.all_products,
            self.product_item_dict,
        )
        self.search_time += time.time() - old_time
        
        # Get product list from search result asins and get list of corresponding URLs
        products = get_product_per_page(top_n_products, page)

        keywords_url_string = '+'.join(keywords)
        url = (
            f'{self.base_url}/search_results/{session_id}/'
            f'{keywords_url_string}/{page}'
        )

        # Render HTML search page and record amount of time taken
        old_time = time.time()
        html = map_action_to_html(
            'search',
            session_id=session_id,
            products=products,
            keywords=session["keywords"],
            page=page,
            total=len(top_n_products),
            instruction_text=session["goal"]["instruction_text"],
        )
        self.render_time += time.time() - old_time
        return html, url
    
    @app.route('/', methods=['GET', 'POST'])
    def item_page(self, session_id, **kwargs):
        """Render and return the HTML for a product item page"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        text_to_clickable = kwargs['text_to_clickable']
        clickable = text_to_clickable[clickable_name]

        # Update session logs with information of last product asin selected
        if (clickable.get('class') is not None and
            clickable.get('class')[0] == 'product-link'):
            session["asin"] = clickable_name.upper()
            session["actions"]["asin"] += 1
            session["asins"].add(session["asin"])
        elif clickable.get('name') is not None:
            clickable_key = clickable['name'].lower()
            session["options"][clickable_key] = clickable_name
            session["actions"]["options"] += 1

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        # Guard against None keywords/page (can happen if session was reset mid-transaction)
        keywords = session.get("keywords") or []
        page = session.get("page") or 1
        keywords_url_string = '+'.join(keywords) if keywords else ''
        option_string = json.dumps(session.get('options') or {})

        url = (
            f'{self.base_url}/item_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/'
            f'{page}/{option_string}'
        )

        html = map_action_to_html(
            'click',
            session_id=session_id,
            product_info=product_info,
            keywords=keywords,
            page=page,
            asin=session["asin"],
            options=session.get("options") or {},
            instruction_text=session["goal"]["instruction_text"],
            show_attrs=self.show_attrs,
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_sub_page(self, session_id, **kwargs):
        """Render and return the HTML for a product's sub page (i.e. description, features)"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        for k in ACTION_TO_TEMPLATE:
            if clickable_name.lower() == k.lower():
                clickable_name = k
                break
        
        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        session["actions"][clickable_name] += 1
        # Guard against None keywords/page (can happen if session was reset mid-transaction)
        keywords = session.get("keywords") or []
        page = session.get("page") or 1
        options = session.get("options") or {}
        keywords_url_string = '+'.join(keywords) if keywords else ''
        url = (
            f'{self.base_url}/item_sub_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/{page}/'
            f'{clickable_name}/{options}'
        )
        html = map_action_to_html(
            f'click[{clickable_name}]',
            session_id=session_id,
            product_info=product_info,
            keywords=keywords,
            page=page,
            asin=session["asin"],
            options=options,
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def done(self, session_id, **kwargs):
        """Render and return HTML for done page"""
        session = self.user_sessions[session_id]
        goal = self.user_sessions[session_id]['goal']
        purchased_product = self.product_item_dict[session["asin"]]
        session["actions"]["purchase"] += 1
        price = self.product_prices.get(session["asin"])

        # Calculate reward for selected product and set variables for page details
        reward, info = get_reward(
            purchased_product,
            goal,
            price=price,
            options=session["options"],
            verbose=True
        )

        self.user_sessions[session_id]['verbose_info'] = info
        self.user_sessions[session_id]['done'] = True
        self.user_sessions[session_id]['reward'] = reward

        url = (
            f'{self.base_url}/done/{session_id}/'
            f'{session["asin"]}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{END_BUTTON}]',
            session_id=session_id,
            reward=reward,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url, reward
    
    def receive(self, session_id, current_url, session_int=None, **kwargs):
        """Map action to the corresponding page"""
        status = dict(reward=0.0, done=False)

        with app.app_context(), app.test_request_context():
            # Create/determine goal, instruction_text from current session
            if session_id not in self.user_sessions:
                idx = session_int if (session_int is not None and isinstance(session_int, int)) else random_idx(self.cum_weights)
                # IMPORTANT: Create a COPY of the goal to avoid modifying shared data
                # The goals list is shared across all SimServer instances for performance,
                # so we must not modify the original goal dictionaries.
                goal = copy.copy(self.goals[idx])
                instruction_text = goal['instruction_text']
                self.user_sessions[session_id] = {'goal': goal, 'done': False}
            else:
                instruction_text = \
                    self.user_sessions[session_id]['goal']['instruction_text']
            if self.assigned_instruction_text is not None:
                instruction_text = self.assigned_instruction_text  # TODO: very hacky, should remove
                self.user_sessions[session_id]['goal']['instruction_text'] = instruction_text
            session = self.user_sessions[session_id]

            if not kwargs:
                # If no action, reset the session variables
                kwargs['instruction_text'] = instruction_text
                html, url = self.index(session_id, **kwargs)
                self.user_sessions[session_id].update(
                    {
                        'keywords': None,
                        'page': None,
                        'asin': None,
                        'asins': set(),
                        'options': dict(),
                        'actions': defaultdict(int)
                    }
                )
            elif 'keywords' in kwargs:
                # If search keywords are available, run a search
                html, url = self.search_results(session_id, **kwargs)
            elif 'clickable_name' in kwargs:
                clickable_name = kwargs['clickable_name'].lower()
                if clickable_name == END_BUTTON.lower():
                    # If "buy now" clicked, calculate reward and flag session as terminated
                    html, url, reward = self.done(session_id, **kwargs)
                    status['reward'] = reward
                    status['done'] = True
                elif clickable_name == BACK_TO_SEARCH.lower():
                    # If "back to search" clicked, ALWAYS return to the initial search page
                    # This allows the user to enter a new search query
                    # Reset session state but preserve the goal
                    self.user_sessions[session_id].update({
                        'keywords': None,
                        'page': None,
                        'asin': None,
                        'asins': set(),
                        'options': dict(),
                        # Note: don't reset 'actions' to track cumulative actions
                    })
                    kwargs_reset = {'instruction_text': instruction_text}
                    html, url = self.index(session_id, **kwargs_reset)
                elif (clickable_name == NEXT_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "next page" clicked from search results, re-render with `page` enumerated
                    # Guard against None page (can happen if session was reset or corrupted)
                    current_page = session.get("page") or 1
                    current_keywords = session.get("keywords") or []
                    if current_keywords:
                        html, url, status = self.receive(
                            session_id,
                            current_url,
                            keywords=current_keywords,
                            page=current_page + 1,
                        )
                    else:
                        # No keywords set, return to search page
                        kwargs_reset = {'instruction_text': instruction_text}
                        html, url = self.index(session_id, **kwargs_reset)
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "prev page" clicked from search results, re-render with `page` denumerated
                    # Guard against None page (can happen if session was reset or corrupted)
                    current_page = session.get("page") or 1
                    current_keywords = session.get("keywords") or []
                    if current_keywords and current_page > 1:
                        html, url, status = self.receive(
                            session_id,
                            current_url,
                            keywords=current_keywords,
                            page=current_page - 1,
                        )
                    elif current_keywords:
                        # Already on page 1, just re-render
                        html, url, status = self.receive(
                            session_id,
                            current_url,
                            keywords=current_keywords,
                            page=1,
                        )
                    else:
                        # No keywords set, return to search page
                        kwargs_reset = {'instruction_text': instruction_text}
                        html, url = self.index(session_id, **kwargs_reset)
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_sub_page'):
                    # If "prev page" clicked from sub page, return to corresponding item page
                    html, url = self.item_page(session_id, **kwargs)
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_page'):
                    # If "prev page" clicked from item page, return to search results page
                    # Guard against None keywords/page
                    current_keywords = session.get("keywords") or []
                    current_page = session.get("page") or 1
                    if current_keywords:
                        html, url = self.search_results(
                            session_id,
                            keywords=current_keywords,
                            page=current_page,
                            **kwargs
                        )
                    else:
                        # No keywords set, return to search page
                        kwargs_reset = {'instruction_text': instruction_text}
                        html, url = self.index(session_id, **kwargs_reset)
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    # Render item_sub_page if clickable is description, features, or reviews
                    html, url = self.item_sub_page(session_id, **kwargs)
                else:
                    # Otherwise, render current item page
                    html, url = self.item_page(session_id, **kwargs)
            return html, url, status
    
    def get_page_name(self, url):
        """Determine which page (i.e. item_page, search_results) the given URL is pointing at"""
        if url is None:
            return None
        page_names = [
            'search_results',
            'item_page',
            'item_sub_page',
            'done'
        ]
        for page_name in page_names:
            if page_name in url:
                return page_name
        return ''  # index page


class SimBrowser:
    """Simulated browser for rendering the HTML source of WebShop environment pages"""
    def __init__(self, server):
        self.server = server
        self.current_url = None
        self.page_source = None
        self.session_id = None

    def get(self, url, session_id=None, session_int=None):
        """Set browser variables to corresponding link, page HTML for URL"""
        self.session_id = url.split('/')[-1] if session_id is None else session_id
        self.page_source, _, _ = \
            self.server.receive(self.session_id, self.current_url, session_int=session_int)
        self.current_url = url
    
    def click(self, clickable_name, text_to_clickable):
        """Wrapper for `receive` handler for performing click action on current page"""
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                clickable_name=clickable_name,
                text_to_clickable=text_to_clickable,
            )
        return status
    
    def search(self, keywords):
        """Wrapper for `receive` handler for performing search action on current page"""
        if isinstance(keywords, str):
            keywords = keywords.split(' ')
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                keywords=keywords,
        )
        return status