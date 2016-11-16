
import sys
import datetime

times = ['10:05', '12:29', '10:48', '13:30', '10:00', '16:12', '6:40', '14:40', '14:51', '14:57', '14:24', '10:12', '10:17', '15:36', '8:57', '14:03', '9:22', '13:05', '17:42', '12:37', '15:03', '9:29', '18:28', '13:51', '12:36', '17:30', '12:18', '9:55', '5:09', '14:19', '16:22', '5:33', '12:31', '16:20', '19:43', '17:25', '18:13', '21:31']

dur = datetime.timedelta()
for i in times:
	t = datetime.datetime.strptime(i, "%M:%S")
	dt = datetime.timedelta(minutes = t.minute, seconds = t.second)
	dur = dur + dt

print(dur)
