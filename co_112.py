"""先做临床，再做基因，接着拿到基因和临床特征，计算临床+基因"""


import pandas as pd

import json
from sklearn.preprocessing import MinMaxScaler

from utils import (
    get_sorted_list_by_data,
    metrics_by_data_top_features,
    metrics_by_data_and_features
)

# 使用argparse获取参数
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
# 我想写一个DROP_CLINICAL_RATE参数，用来控制删除临床指标中缺失值过多的列比例，比如0.1，就是删除缺失值超过总样本数10%的列
parser.add_argument('--DROP_CLINICAL_RATE', type=float, default=1.0, help='删除临床指标中缺失值过多的列比例')
parser.add_argument('--debug', type=bool, default=False, help='是否是debug模式')


if __name__ == "__main__":
    raw_clinical_path = "ukb/co_data/临床数据_filter.xlsx"
    raw_gene_path = "ukb/co_data/基因数据.xlsx"
    co_path = "ukb/co_data/协变量.xlsx"

    args = parser.parse_args()
    if args.debug:
        print("debug模式")
        raw_gene_df = pd.read_excel(raw_gene_path).set_index("ID").iloc[:1000,:]
        raw_clinical_df = pd.read_excel(raw_clinical_path).set_index("ID").iloc[:1000,:]
        co_df = pd.read_excel(co_path).set_index("ID").iloc[:1000,:]

    else:
        raw_gene_df = pd.read_excel(raw_gene_path).set_index("ID")
        raw_clinical_df = pd.read_excel(raw_clinical_path).set_index("ID")
        co_df = pd.read_excel(co_path).set_index("ID")


    DROP_CLINICAL_RATE = parser.parse_args().DROP_CLINICAL_RATE
    print("删除临床指标中缺失值过多的列比例为：", DROP_CLINICAL_RATE)
    # 计算每列的缺失值数量
    missing_values = raw_clinical_df.isnull().sum()

    # 计算总样本数
    total_samples = len(raw_clinical_df)

    # 计算阈值，样本缺失个数多于总样本的rate的列将被删除
    threshold = DROP_CLINICAL_RATE * total_samples

    # 筛选出要删除的列
    columns_to_drop = missing_values[missing_values >= threshold].index

    # 删除列
    cleaned_clinical_df = raw_clinical_df.drop(columns=columns_to_drop)

    # 修改为列名为中文
    cleaned_clinical_df.rename(columns={"收缩压平均值": "Systolic blood pressure average"}, inplace=True)

    # 拿到数据和标签
    clinical_df = cleaned_clinical_df.iloc[:,1:]
    clinical_label = cleaned_clinical_df["表型"]


    import time
    

    # 临床模型
    print("临床模型开始进行")
    start_time = time.time()
    
    # 临床数据归一化
    scaler = MinMaxScaler()

    # 合并临床和协变量
    clinical_df = pd.concat([clinical_df, co_df], axis=1)
    print("clinical_df.shape: ", clinical_df.shape)

    clinical_normalized = pd.DataFrame(scaler.fit_transform(clinical_df), columns=clinical_df.columns)

    clinical_sorted_dict, clinical_top_features, clinical_name_importance = \
        get_sorted_list_by_data(clinical_normalized, clinical_label)
    
    clinical_num_feature = clinical_sorted_dict[0]['features_numbers']

    # 保存结果目录
    import os
    save_path = f"co_result/normalized_112/{DROP_CLINICAL_RATE}"
    if args.debug:
        save_path = f"co_result/debug/{DROP_CLINICAL_RATE}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存临床features和对应的roc_auc
    with open(os.path.join(save_path, "clinical_features_roc_auc.json"), "w") as f:
        json.dump({"features_roc": clinical_sorted_dict, 
                   "top_features": clinical_top_features}, f)

    clinical_roc_auc_list, clinical_model_metrics = \
        metrics_by_data_top_features(clinical_normalized, clinical_label, clinical_top_features, clinical_num_feature, roc_auc_list=[])

    # 临床处理完毕，保存model_metrics结果,用json保存
    with open(os.path.join(save_path, "clinical_model_metrics.json"), "w") as f:
        json.dump({"clinical_model_metrics": clinical_model_metrics, "clinical_model_shap": clinical_name_importance}, f)

    # 临床模型处理完毕，保存roc_auc_list结果,用json保存
    with open(os.path.join(save_path, "clinical_roc_auc_list.json"), "w") as f:
        json.dump(clinical_roc_auc_list, f)
    print(f"clinical_roc_auc_list save success, time: {(time.time() - start_time) / 60}min")

    # 保存临床模型中使用的特征个数，以及特征
    clinical_features_dict = {"clinical_len": clinical_num_feature, "clinical_features": list(clinical_top_features[:clinical_num_feature])}


    # 基因模型
    start_time = time.time()

    # 拿到数据和标签
    gene_df = raw_gene_df.iloc[:,1:]
    gene_label = raw_gene_df["表型"]

    # 基因数据归一化
    scaler = MinMaxScaler()

    # 合并基因和协变量
    gene_df = pd.concat([gene_df, co_df], axis=1)
    print("gene_df.shape: ", gene_df.shape)

    gene_normalized = pd.DataFrame(scaler.fit_transform(gene_df), columns=gene_df.columns)

    gene_sorted_dict, gene_top_features, gene_name_importance = \
        get_sorted_list_by_data(gene_normalized, gene_label)
    
    gene_num_feature = gene_sorted_dict[0]['features_numbers']

    # 保存基因features和对应的roc_auc
    with open(os.path.join(save_path, "gene_features_roc_auc.json"), "w") as f:
        json.dump({"features_roc": gene_sorted_dict, 
                   "top_features": gene_top_features}, f)

    gene_roc_auc_list, gene_model_metrics = \
        metrics_by_data_top_features(gene_normalized, gene_label, gene_top_features, gene_num_feature, roc_auc_list=[])
    
    # 基因处理完毕，保存model_metrics结果,用json保存
    with open(os.path.join(save_path, "gene_model_metrics.json"), "w") as f:
        json.dump({"gene_model_metrics": gene_model_metrics, "gene_model_shap": gene_name_importance}, f)

    # 基因模型处理完毕，保存roc_auc_list,用json保存
    with open(os.path.join(save_path, "gene_roc_auc_list.json"), "w") as f:
        json.dump(gene_roc_auc_list, f)
    print(f"gene_roc_auc_list save success, time: {(time.time() - start_time) / 60}min")

    # 保存基因模型中使用的特征个数，以及特征
    gene_features_dict = {"gene_len": gene_num_feature, "gene_features": list(gene_top_features[:gene_num_feature])}



    start_time = time.time()
    # 临床+基因数据处理

    # 合并

    # 去除clinical_normalized中的co_feature列
    co_feature = "duration_of_diabetes"
    clinical_normalized.drop(columns=co_feature, inplace=True)
    gene_clinical_normalized = pd.concat([gene_normalized, clinical_normalized], axis=1)
    print("gene_clinical_normalized.shape: ", gene_clinical_normalized.shape)

    gene_clinical_label = gene_label

    # 拼接临床和基因的特征
    gene_clinical_features = list(set(list(gene_top_features[:gene_num_feature]) + list(clinical_top_features[:clinical_num_feature])))

    # 如果co_feature不在gene_clinical_features中，就加入
    if co_feature not in gene_clinical_features:
        gene_clinical_features.append(co_feature)


    # 临床+基因模型
    print("临床+基因模型开始进行")


    # gene_clinical_sorted_dict, gene_clinical_top_features, gene_clinical_name_importance = \
    #     get_sorted_list_by_data(gene_clinical_normalized, gene_clinical_label)
    
    # 临床+基因 各个模型概率和roc_auc的保存目录
    gene_clinical_probabilities_acc_path = os.path.join(save_path, "gene_clinical_probabilities_acc")
    gene_clinical_roc_auc_list, gene_clinical_model_metrics = metrics_by_data_and_features(gene_clinical_normalized, 
                                                                                               gene_clinical_label,
                                                                                               gene_clinical_features,
                                                                                                roc_auc_list=[])
    # 保存临床+基因模型各个模型概率和roc_auc
    with open(os.path.join(save_path, "gene_clinical_model_metrics.json"), "w") as f:
        json.dump(gene_clinical_model_metrics, f)


    gene_clinical_len = len(gene_clinical_features)

    # 保存临床基因模型中使用的特征个数，以及特征
    gene_clinical_features_dict = {"gene_clinical_len": gene_clinical_len, "gene_clinical_features": gene_clinical_features}

    
        
    # 临床+基因处理完毕，保存结果,用json保存
    import json
    with open(os.path.join(save_path, "gene_clinical_roc_auc_list.json"), "w") as f:
        json.dump(gene_clinical_roc_auc_list, f)
    
    print(f"gene_clinical_roc_auc_list save success, time: {(time.time() - start_time) / 60}min")


    # 汇总gene_clinical_features_dict, clinical_features_dict, gene_features_dict,把三个字典合并成一个字典
    all_features_dict = {"gene_clinical_features_dict": gene_clinical_features_dict, "clinical_features_dict": clinical_features_dict, "gene_features_dict": gene_features_dict}
    with open(os.path.join(save_path, "all_features_dict.json"), "w") as f:
        json.dump(all_features_dict, f)

    # 汇总结果
    all_dict = {"clinical": clinical_roc_auc_list,
            "gene": gene_roc_auc_list,
            "clinical_gene": gene_clinical_roc_auc_list}
    final_dict = {}
    for name, roc_auc_list in all_dict.items():
        models = []
        roc_auc_mean = []
        for dic in roc_auc_list:
            models.append(dic["model"])
            roc_auc_mean.append(dic["roc_auc_mean"])
        final_dict[name] = {"models": models, "roc_auc_mean": roc_auc_mean}


    df_index = final_dict["clinical"]["models"]
    df_data = {}
    for key, value in final_dict.items():
        df_data[key] = value["roc_auc_mean"]
    final_df = pd.DataFrame(data=df_data, index=df_index)

    # 保存最终结果
    final_df.to_csv(os.path.join(save_path, "final_result.csv"))
    print("done!")




    

