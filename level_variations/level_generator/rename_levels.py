# importing os module
import os
from shutil import copy

# level_base_dir = '../generated_levels/third generation - 200 levels filtered to 100/'
level_base_dir = '../generated_levels/fourth generation -  20 levels for heuristic testing/'
level_list_to_delete = 'levels_to_delete.csv'


# Function to rename multiple files
def main():
	for hierarchy_level in os.listdir(level_base_dir):
		for capability in os.listdir(level_base_dir + hierarchy_level):
			for template in os.listdir(level_base_dir + hierarchy_level + '/' + capability):
				print(level_base_dir + hierarchy_level + '/' + capability + '/' + template)
				i = 1
				for level in os.listdir(level_base_dir + hierarchy_level + '/' + capability + '/' + template):
					# only keep 100 levels
					if i > 100:
						print('deleting level:', level_base_dir + hierarchy_level + '/' + capability + '/' + template + '/' + level)
						os.remove(level_base_dir + hierarchy_level + '/' + capability + '/' + template + '/' + level)
						continue

					print('renaming level:', level_base_dir + hierarchy_level + '/' + capability + '/' + template + '/' + level)
					os.rename(level_base_dir + hierarchy_level + '/' + capability + '/' + template + '/' + level,
							  level_base_dir + hierarchy_level + '/' + capability + '/' + template + '/' + level.split('_')[0] + '_' + level.split('_')[1] + '_' + level.split('_')[
								  2] + '_' + "{0:05d}".format(i) + '.xml')
					i += 1
	# 		break
	# 	break
	# break


# i = 1
# "{0:05d}".format(i)
# for filename in os.listdir(source_dir):
# 	# new_file_name = filename.split('_')[0] + '_'+ filename.split('_')[1] + '_0_7_3'
# 	new_file_name = filename.split('_')[0] + '_1_0_7_3'
# 	print(new_file_name)
#
# 	dst = destination_dir + new_file_name + ".xml"
# 	src = source_dir + '/' + filename
#
# 	os.rename(src, dst)
# 	i += 1


# Driver Code
if __name__ == '__main__':
	# Calling main() function
	main()
