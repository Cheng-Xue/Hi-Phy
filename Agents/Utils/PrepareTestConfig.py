import os
from shutil import copyfile

import lxml.etree as etree
os.chdir("../")
operating_system = 'Linux'
target_level_path = '../level_variations/generated_levels/fourth generation'
origin_level_path = '../buildgame/{}/9001_Data/StreamingAssets/Levels/novelty_level_1/type1/Levels/'.format(operating_system)
game_level_path = '9001_Data/StreamingAssets/Levels/novelty_level_1/type1/Levels/'.format(operating_system)
game_config_path = '../buildgame/{}/config.xml'.format(operating_system)

# each for each template, move 20 levels
hi_levels = os.listdir(target_level_path)
# remove all the levels in self.origin_level_path
old_levels = os.listdir(origin_level_path)
for old_level in old_levels:
    os.remove(os.path.join(origin_level_path, old_level))

total_template_level_path = []
for level in hi_levels:
    capabilites = os.listdir(os.path.join(target_level_path,level))
    for capability in capabilites:
        templates = os.listdir(os.path.join(target_level_path,level,capability))
        for template in templates:
            game_levels = os.listdir(os.path.join(target_level_path,level,capability,template))
            for game_level in game_levels[:20]:
                src_path = os.path.join(target_level_path,level,capability,template,game_level)
                dst_path = os.path.join(origin_level_path, game_level)
                copyfile(src_path, dst_path)
                total_template_level_path.append(os.path.join(game_level_path, game_level))


parser = etree.XMLParser(encoding='UTF-8')
game_config = etree.parse(game_config_path, parser=parser)
config_root = game_config.getroot()
# remove old level path
for level in list(config_root[1][0][0]):
    config_root[1][0][0].remove(level)
# add new level path
for l in total_template_level_path:
    new_level = etree.SubElement(config_root[1][0][0], 'game_levels')
    new_level.set('level_path', l)

# add a repeated level for the weird not loadding last level bug
new_level = etree.SubElement(config_root[1][0][0], 'game_levels')
new_level.set('level_path', l)

game_config.write(game_config_path)


