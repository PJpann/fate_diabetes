*此为项目脚本文件说明，对于代码有如下几点说明：
 1）脚本文件为.py文件，所运行代码均为python语言编写，且运行前最好保证有pycharm IDE且保证版本python3.6以上。
2）先下载setuptools包，然后进入cmd，进入setup.py所在文件的位置，输入“python setup.py install”安装相关包和依赖。如若权限不够，输入“python setup.py install --user”
3）下载完后，原始数据文件为big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0.zip，解压后，运行Feature_engineering.py，输入数据为题目所给出的数据文件，要进行复现的话，必须保证给出的数据和所给的数据格式一致，确保文件目录一致,输出构建完成特征后的用户数据后，运行feature_selection进行特征的筛选，筛选出20条特征。
4）fate的版本最好为1.11.4，通过split.py随机按9:1的比例多次划分16个用户数据和生成测试数据，生成数据文件split_folder_xxx(xxx为随机种子)，上传数据文件，upload_userx.json(x为不同的用户)，如upload_user1.json;上传数据文件命令： flow data upload -c upload_user3.json
5)编写dsl.json和conf.json文件，提交训练和验证任务。
 提交任务命令：flow job submit -d homosecureboostmultihost_dsl.json -c  homosecureboost_multihost_conf.json
6)联邦学习配置文件：homosecureboost_dsl.json; homosecureboost_multihost_conf.json
7)中心学习化学习的配置文件：homosecureboost_dsl.json;homosecureboost_conf.json
