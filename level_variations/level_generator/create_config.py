import os

level_input_folder = 'C:/Users/u7068760/Desktop/Benchmark/buildgame/Windows/9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels/'

# read levels in the folder

level_files = os.listdir(level_input_folder)
print('level_files', level_files)

config_file = open('config.xml', "w")

# write the headers
for level_file in level_files:
	config_file.write('        <game_levels level_path="' + level_input_folder + level_file + '" />\n')
