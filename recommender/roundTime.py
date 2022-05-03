from datetime import datetime

def roundTime(dt=None, roundTo=15):
	"""Round a datetime object to any time lapse in seconds
	dt : datetime.datetime object, default now.
	roundTo : Closest number of seconds to round to, default 1 minute.
	"""
	if dt == None : dt = datetime.now()
	seconds = dt.replace(second=0,microsecond=0).timestamp()
	remainder = seconds % (roundTo * 60)
	return seconds - remainder

def getNowStr():
	now = datetime.now()
	nowStr = now.strftime("%Y%m%d-%H%M")
	return nowStr

if __name__ == "__main__":
	datetime_str = "20220112-2005"
	date_time_obj = datetime.strptime(datetime_str, '%Y%m%d-%H%M')

	latest_15_round_ts = roundTime(date_time_obj)
	dt_object = datetime.fromtimestamp(latest_15_round_ts)

	print("orginal_dt_object=", date_time_obj)
	print("dt_round_object =", dt_object)

	print("Original timestamp: ", date_time_obj.timestamp())
	print("15 minutes rounding timestamp: ", latest_15_round_ts)

	print("Now:", getNowStr())