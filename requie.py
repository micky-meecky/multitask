# 读取txt文件，并将文件转化为requirements.txt格式, 因为他这个是包含了conda和pip的，所以我需要将他分开
# 后来发现这是在linux平台下的，所以不需要分开，在Linux下，直接conda create --name xxx --file requirements.txt就可以了
# 导入模块
import os

# 定义函数，读取TXT文件
def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()   # 读取所有行，这个lines是一个列表
        return lines

# 定义函数，将读取的文件转化为requirements.txt格式
def extract_file(lines, file_name = r'./requirements/'):   # lines是一个列表，每个元素是一行，每行是一个字符串，file_name是用来存放转化后的文件的路径
    # 源文件格式为augmentor=0.2.2=pypi_0
                # bleach=3.1.0=py_0
                # cairo=1.14.12=h8948797_3
    # 这样的，我需要识别最后一个等号后的内容,如果是pypi_0,则将这行放在另外一个名为pypi.txt的文件中
    # 如果不是pypi_0,则将这行放在另外一个名为conda.txt文件中

    # 定义两个列表，分别存放pypi.txt和conda.txt
    pypi_list = []
    conda_list = []

    # 用for循环遍历列表中的每个元素，用split函数将每个元素按照等号分割，得到一个列表
    for line in lines:
        line_list = line.split('=')  # line_list是一个列表，每个字符串是等号分割后的一个元素，
        # 如果line_list的最后一个元素是以py开头的，则将这一行放在pypi.txt中
        if line_list[-1].startswith('pypi') or line_list[-1].startswith('py36_') or line_list[-1].startswith('py_'):
            # 从line中截取最后一个等号前的内容，即为包名
            package_name = line[:line.rfind('=')]   # rfind函数是从右边开始查找，找到第一个等号的位置，然后截取从0到这个位置的字符串
            # 现在将这个包名的一个等号替换为两个等号
            package_name = package_name.replace('=', '==')   # replace函数是将字符串中的一个等号替换为两个等号
            # 将这个包名放在pypi.txt中
            pypi_list.append(package_name)
        else:
            # 将这行原封不动的放在conda.txt中
            conda_list.append(line)

    # 将pypi_list和conda_list写入到pypi.txt和conda.txt中
    # 写入pypi.txt
    with open(file_name + 'pypi.txt', 'w') as f:
        f.write('\n'.join(pypi_list))

    # 写入conda.txt
    with open(file_name + 'conda.txt', 'w') as f:
        f.write(''.join(conda_list))

    print('转化完成')


# 定义一个函数，用于统计txt文件的行数，以及等号的个数，还有判断是否每一行都有且仅有两个等号
def Count(lines):
    # 定义一个变量，用于统计行数
    count = 0
    # 定义一个变量，用于统计等号的个数
    equal_count = 0
    # 定义一个变量，用于统计每一行都有且仅有两个等号
    two_equal = 0

    # 获取列表长度，即为行数
    count = len(lines)

    # 获取等号的个数
    for line in lines:
        equal_count += line.count('=')

    # 判断每一行都有且仅有两个等号
    for line in lines:
        line_list = line.split('=')
        if len(line_list) == 3: # 如果每一行都有且仅有两个等号，则每个元素的个数为3
            two_equal += 1  # 如果这一行有且仅有两个等号，则two_equal加1
        else:
            # 将改行打印出来
            print('有问题的行：', line)

    # 将行数与总等于号数作比较，如果行数等于等于号的1/2，说明满足条件，总等于号 = 行数 * 2的条件，并输出这句话
    if count == equal_count/2:
        print('行数 * 2 = 等号数')
    else:
        print('有问题')

    # 将行数与two_equal作比较，如果相等，则说明每一行都有且仅有两个等号
    if count == two_equal:
        print('每一行都有且仅有两个等号')
    else:
        print('有问题')

    # 输出行数，等号数，以及每一行都有且仅有两个等号的个数
    print('行数：', count)
    print('等号数：', equal_count)
    print('有且仅有两个等号的个数：', two_equal)


# main
if __name__ == '__main__':
    # 文件名字
    file_name = r'./requirements.txt'  # r表示原始字符串，不转义
    # 保存的文件地址
    save_path = r'./requirements/'
    # 读取文件
    lines = read_file(file_name)    # lines是一个列表，每个元素是一行，每行是一个字符串
    # 统计文件
    Count(lines)
    # 转化文件
    extract_file(lines, save_path)









