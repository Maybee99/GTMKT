import EduData
import pandas as pd
import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 教育数据集名字
data_name_list = ['assistment-2009-2010-skill', 'assistment-2012-2013-non-skill', 'assistment-2015',
                  'assistment-2017', 'junyi', 'KDD-CUP-2010', 'NIPS-2020', 'slepemapy.cz', 'synthetic',
                  'psychometrics', 'psy', 'pisa2015', 'workbankr', 'critlangacq', 'ktbd', 'ktbd-a0910',
                  'ktbd-junyi', 'ktbd-synthetic', 'ktbd-a0910c', 'cdbd', 'cdbd-lsat', 'cdbd-a0910',
                  'math2015', 'ednet', 'ktbd-ednet', 'math23k', 'OLI-Fall-2011', 'open-luna']


def Dataset_load(dataset):
    # 下载数据
    data = EduData.DataSet.get_data(dataset=dataset, data_dir="./Data_Edu/raw")
    # 获取下载文件夹下的文件名
    data_file_name = os.listdir(data)[0]
    # 文件路径
    data_file = os.path.join(data, data_file_name)
    # 读取文件
    data = pd.read_csv(data_file, encoding="ISO-8859-15", low_memory=False)

    return data


# junyi教育数据集
data_name = "junyi"


class Edu_dataset_junyi(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Edu_dataset_junyi, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['edu_dataset.dataset']

    def download(self):
        EduData.DataSet.get_data(dataset=data_name, data_dir="./Data_Edu/raw")  # 下载数据集

    def process(self):
        # 写入保存数据的方法函数
        data_list = []  # Read data into huge `Data` list.
        # junyi教育数据集加载
        junyi_exercise = pd.read_csv("./Data_Edu/raw/junyi/junyi_Exercise_table.csv", encoding="utf-8",
                                     low_memory=False)
        junyi_problemlog_original = pd.read_csv("./Data_Edu/raw/junyi/junyi_ProblemLog_original.csv", encoding="utf-8")
        relationship_exercise = pd.read_csv("./Data_Edu/raw/junyi/relationship_annotation_training.csv",
                                            encoding="utf-8")
        merged_df = pd.merge(junyi_problemlog_original, junyi_exercise, left_on='exercise', right_on='name', how='left')

        pd.set_option('display.max_rows', None)  # 设置显示所有行
        item_encoder = LabelEncoder()

        # 将junyi_Exercise_table中的name编码映射到relationship_annotation_training中
        junyi_exercise = junyi_exercise.drop_duplicates("name").reset_index(drop=True)
        junyi_exercise["encoder"] = item_encoder.fit_transform(junyi_exercise["name"])

        relationship_exercise["Exercise_A"] = relationship_exercise["Exercise_A"].map(
            junyi_exercise.set_index("name")["encoder"])
        relationship_exercise["Exercise_B"] = relationship_exercise["Exercise_B"].map(
            junyi_exercise.set_index("name")["encoder"])

        merged_df["exercise"] = item_encoder.fit_transform(merged_df["exercise"])
        merged_df["name"] = item_encoder.fit_transform(merged_df["name"])
        merged_df["topic"] = item_encoder.fit_transform(merged_df["topic"])
        merged_df["area"] = item_encoder.fit_transform(merged_df["area"])
        merged_df["prerequisites"] = item_encoder.fit_transform(merged_df["prerequisites"])

        # print(merged_df)
        """
        构建图数据集
        """
        # 按照学生user_id分类
        sampled_user_id = np.random.choice(merged_df["user_id"].unique(), 1000, replace=False)
        merged_df = merged_df.loc[merged_df["user_id"].isin(sampled_user_id)]
        grouped = merged_df.groupby("user_id")
        # print("grouped_len:", len(grouped))

        # 构造图数据
        for user_id, group in tqdm(grouped):
            # print("user_id: ", user_id)

            exercise_info_list = group.to_dict(orient='records')

            for exercise_info in exercise_info_list:
                # 节点特征
                node_features = []
                # 边信息
                target_nodes = []
                source_nodes = []

                # print(exercise_info)
                correct = exercise_info["correct"]
                # print("correct:", correct)

                # 一级节点 学生-练习：1-1
                source_nodes.append(0)
                target_nodes.append(1)
                node_features.append(
                    [exercise_info["user_id"], exercise_info["count_attempts"], float(exercise_info["count_hints"]),
                     float(exercise_info["points_earned"])])  # 学生节点的特征
                node_features.append([exercise_info["exercise"], exercise_info["prerequisites"], exercise_info["topic"],
                                      exercise_info["area"]])  # 练习节点的特征
                # print("node_features:", node_features)

                # 二级节点
                exercise_id = exercise_info["exercise"]
                # print("exercise_id:", exercise_id)
                relationship_exercise_info = relationship_exercise[relationship_exercise["Exercise_A"] == exercise_id]
                relationship_exercise_info = relationship_exercise_info.to_dict(orient='records')

                if relationship_exercise_info:
                    # 二级节点 练习-概念：1-N
                    relationship_exercise_info_len = len(relationship_exercise_info)
                    for rela_exercise, relationship_exercise_index in zip(relationship_exercise_info,
                                                                          range(2, relationship_exercise_info_len)):
                        # print("rela_exercise:", rela_exercise)
                        # 边信息
                        source_nodes.append(1)
                        target_nodes.append(relationship_exercise_index)
                        # 节点特征
                        node_features.append([rela_exercise["Exercise_B"], float(rela_exercise["Similarity_avg"]),
                                              float(rela_exercise["Difficulty_avg"]),
                                              float(rela_exercise["Prerequisite_avg"])])

                    # ----------------------------------------------图数据
                    # edge_index
                    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                    # X
                    X = torch.tensor(node_features, dtype=torch.float)
                    # y
                    y = torch.FloatTensor([correct])
                    # 图数据
                    data = Data(x=X, edge_index=edge_index, y=y)
                    # print("data:", data)
                    if self.pre_filter is not None:
                        data_list = [data for data in data_list if self.pre_filter(data)]
                    if self.pre_transform is not None:
                        data_list = [self.pre_transform(data) for data in data_list]

                    data_list.append(data)
                    data, slices = self.collate(data_list)
                    torch.save((data, slices), self.processed_paths[0])
                else:
                    # ----------------------------------------------图数据
                    # edge_index
                    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                    # X
                    X = torch.tensor(node_features, dtype=torch.float)
                    # y
                    y = torch.FloatTensor([correct])
                    # 图数据
                    data = Data(x=X, edge_index=edge_index, y=y)
                    # print("data:", data)
                    if self.pre_filter is not None:
                        data_list = [data for data in data_list if self.pre_filter(data)]
                    if self.pre_transform is not None:
                        data_list = [self.pre_transform(data) for data in data_list]

                    data_list.append(data)

                    data, slices = self.collate(data_list)
                    torch.save((data, slices), self.processed_paths[0])

