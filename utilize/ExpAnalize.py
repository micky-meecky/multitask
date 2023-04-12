import os
import pandas as pd

class ExperimentAnalyzer:
    def __init__(self, savemodel_dir, exp_prefix, n_rows):
        self.savemodel_dir = savemodel_dir
        self.exp_prefix = exp_prefix
        self.n_rows = n_rows
        self.exp_groups = []

    def find_exp_groups(self):
        exp_dirs = os.listdir(self.savemodel_dir)
        exp_groups = []
        for exp_dir in exp_dirs:
            # 还要判断是否是文件夹，如果不是文件夹，就不要加入exp_groups
            if os.path.isdir(os.path.join(self.savemodel_dir, exp_dir)) and \
                    exp_dir.startswith(self.exp_prefix):
                exp_groups.append(exp_dir)

            # if exp_dir.startswith(self.exp_prefix):
            #     exp_groups.append(exp_dir)
        self.exp_groups = exp_groups

    def analyze_exp_group(self, exp_group):
        record_file = os.path.join(self.savemodel_dir, exp_group, 'record.xlsx')
        df = pd.read_excel(record_file, engine='openpyxl', header=1)
        headers = list(df.columns[3:])
        avg_last_n_rows = df.tail(self.n_rows)[headers].mean()
        max_iou_row = df.nlargest(1, 'IOU')[headers]
        # print 他们俩
        avg_last_n_rows = avg_last_n_rows.to_frame().transpose()
        print(avg_last_n_rows)
        print(max_iou_row)
        # avg_results = pd.concat([avg_last_n_rows.transpose(), max_iou_row])
        # avg_exp = avg_results.mean()
        return avg_last_n_rows, max_iou_row, headers

    def analyze_all(self):
        self.find_exp_groups()
        results = []
        for exp_group in self.exp_groups:
            avg_last_n_rows, max_iou_row, headers = self.analyze_exp_group(exp_group)
            results.append(avg_last_n_rows)
            results.append(max_iou_row)
        results = pd.concat(results)
        df = pd.DataFrame(results, columns=headers)
        return df


class ResultSaver:
    def __init__(self, savemodel_dir, exp_prefix, n_rows):
        self.savemodel_dir = savemodel_dir
        self.exp_prefix = exp_prefix
        self.n_rows = n_rows
        self.analyzer = ExperimentAnalyzer(savemodel_dir, exp_prefix, n_rows)
        self.avg_avg_path = os.path.join(savemodel_dir, exp_prefix, 'avg_avg')
        self.max_avg_path = os.path.join(savemodel_dir, exp_prefix, 'max_avg')

    def run(self):
        df = self.analyzer.analyze_all()
        even_rows = df.iloc[::2]    # even行的就是avg_last_n_rows
        odd_rows = df.iloc[1::2]    # odd行的就是max_iou_row
        print('haha ')
        even_rows = even_rows.apply(lambda x: pd.to_numeric(x, errors='coerce'))  # 转换数据类型为浮点型
        odd_rows = odd_rows.apply(lambda x: pd.to_numeric(x, errors='coerce'))  # 转换数据类型为浮点型
        # avg_even_rows = even_rows.mean(axis=0)  # 按列求平均
        avg_avg = even_rows.mean(axis=0)  # 按列求平均
        max_avg = odd_rows.mean(axis=0)
        # 检测avg_avg_path和max_avg_path是否存在，如果不存在就创建
        if not os.path.exists(self.avg_avg_path):
            os.makedirs(self.avg_avg_path)
        if not os.path.exists(self.max_avg_path):
            os.makedirs(self.max_avg_path)

        # 保存avg_avg和max_avg
        avg_avg_file = os.path.join(self.avg_avg_path, 'avg_avg.txt')
        max_avg_file = os.path.join(self.max_avg_path, 'max_avg.txt')
        with open(avg_avg_file, 'w') as f:
            f.write(str(avg_avg))
        with open(max_avg_file, 'w') as f:
            f.write(str(max_avg))


if __name__ == '__main__':
    savemodel_dir = r'../savemodel/'
    exp_prefix = 'unet_04_f'
    n_rows = 20
    saver = ResultSaver(savemodel_dir, exp_prefix, n_rows)
    saver.run()
