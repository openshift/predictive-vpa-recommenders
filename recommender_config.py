import yaml

config_file = '/predictive-vpa-recommenders/config/recommender_config.yaml'
config = yaml.load(open(config_file,"r"), Loader=yaml.FullLoader)

# Retrieve the configuration for the recommender
RECOMMENDER_NAME = config['RECOMMENDER_NAME']
DEFAULT_NAMESPACE = config['DEFAULT_NAMESPACE']
PROM_URL=config['PROM_URL']
PROM_TOKEN=config['PROM_TOKEN']

# Retrieve the configuration for the recommendation algorithm
TREE = config['tree']
TARGET_PERCENTILE = config['TARGET_PERCENTILE']
UPPERBOUND_PERCENTILE = config['UPPERBOUND_PERCENTILE']
LOWERBOUND_PERCENTILE = config['LOWERBOUND_PERCENTILE']
LIMIT = config['LIMIT']
OUTPUT = config['OUTPUT']
STAT_THRESHOLD = config['STAT_THRESHOLD']
THETA_THRESHOLD = config['THETA_THRESHOLD']
WINDOW_SPILLTER = config['WINDOW_SPLITTER']
MAX_CHANGEPOINTS = config['MAX_CHANGEPOINTS']
FORECASTING_WINDOW = config['FORECASTING_WINDOW']
FORECASTING_SIGHT = config['FORECASTING_SIGHT']
SAMPLING_PERIOD = config['SAMPLING_PERIOD']
DEFAULT_UTILIZATION = config['DEFAULT_UTILIZATION']
SLEEP_WINDOW = config['SLEEP_WINDOW']
