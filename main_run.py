
import os
for i in range(0, 17):
	print("the target index is", i)
	over = os.system("python Infor_mine.py --target_index=" + str(i))