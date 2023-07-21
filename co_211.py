import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from utils import (
    get_sorted_list_by_data,
    metrics_by_data_top_features,
    metrics_by_data_and_features
)

# MAX_ITER = 8000
# def get_sorted_list_by_data(data, label):
#     """输入数据和标签，得到roc_auc最好前n个特征"""
#     model = LogisticRegression(max_iter=MAX_ITER)
#     features = data.columns
#     train_data = pd.DataFrame(KNN(k = 10).fit_transform(data), columns=features)
#     model.fit(train_data, label)
#     explainer = shap.LinearExplainer(model, train_data)

#     shap_values = explainer(train_data)
#     # 获取特征重要性
#     feature_importances = np.abs(shap_values.values).mean(0)
#     feature_names = train_data.columns

#     # 获取特征名称和对应的重要性
#     name_importance = {}
#     for name, importance in zip(feature_names, feature_importances):
#         name_importance[name] = importance

#     # 排序
#     sorted_idx = feature_importances.argsort()[::-1]
#     top_features = [feature_names[i] for i in sorted_idx]
#     # summarize the effects of all the features
#     # 绘制特征重要性条形图并获取特征重要性数据
#     # shap.summary_plot(shap_values, train_data, plot_type='bar', max_display=20)


#     # 对前面的特征进行排序训练

#     kflod = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
#     scoring = ['accuracy','roc_auc']
#     roc_auc_list = []
#     for i in range(1,len(top_features) + 1):
#         # xgb = xgboost.XGBClassifier()
#         imputer = ColumnTransformer([
#                 ('knn_imputer', KNNImputer(n_neighbors=10), top_features[:i]),
#             ])
#         model = LogisticRegression(max_iter=MAX_ITER)
#         pipeline = Pipeline([
#                 ('imputer', imputer),
#                 ('classifier', model),
#             ])
#         scores = cross_validate(pipeline, data.loc[:,top_features[:i]], label,
#                                 cv=kflod, scoring=scoring)
#         acc = np.array(scores['test_accuracy'])
#         roc_auc = np.array(scores['test_roc_auc'])
#         print("features_numbers: ", i)
#         # print("roc_auc mean: ", roc_auc.mean(), "roc_auc std: ", roc_auc.std(), "roc_auc max:", roc_auc.max())
#         roc_auc_list.append({'features_numbers': i, "roc_auc_mean": roc_auc.mean(), "roc_auc": list(roc_auc)})
#         # print("="*80)

#     # 对roc_auc_list进行排序
#     sorted_dict = sorted(roc_auc_list, key=lambda x:x['roc_auc_mean'], reverse=True)
#     return sorted_dict, top_features, name_importance

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier

# def metrics_by_data_top_features(data, label, top_features, num_feature, roc_auc_list):
#     """
#     save_path: 保存预测概率的路径
#     """
#     kflod = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
#     #定义模型
#     models = [LogisticRegression(max_iter=MAX_ITER), KNeighborsClassifier(), SVC(probability=True), RandomForestClassifier(), MLPClassifier(hidden_layer_sizes=(5,), max_iter=500), xgboost.XGBClassifier()]
#     models_metrics = []
#     for model in models:
#         imputer = ColumnTransformer([
#                 ('knn_imputer', KNNImputer(n_neighbors=10), top_features[:num_feature]),
#             ])
#         pipeline = Pipeline([
#                 ('imputer', imputer),
#                 ('classifier', model),
#             ])
#         # scores = cross_validate(pipeline, data.loc[:,top_features[:num_feature]], label, 
#         #                         cv=kflod, scoring=scoring)
        
#         # 存储每个样本的预测概率
#         probabilities = []
#         all_prob = {}
#         accuracies = []
#         roc_aucs = []
#         # 五折交叉验证
#         X = data.loc[:,top_features[:num_feature]]
#         for train_index, test_index in kflod.split(X, label):
#             X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
#             y_train, y_test = label.iloc[train_index], label.iloc[test_index]

#             pipeline.fit(X_train, y_train)

#             # 预测概率
#             y_prob = pipeline.predict_proba(X_test)[:,1]
#             idx = list(y_test.index)
            
#             # id和概率对应
#             for id, prob in zip(idx, y_prob):
#                 all_prob[id] = float(prob)

#             probabilities.append((y_test.values, y_prob))

#             # 计算roc_auc
#             roc_auc = roc_auc_score(y_test, y_prob)
#             roc_aucs.append(roc_auc)

#             # 预测类别
#             y_pred = pipeline.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             accuracies.append(accuracy)
        
#         y_test_list = []
#         y_prob_list = []
#         for probability in probabilities:
#             t, p = probability
#             y_test_list.append(t)
#             y_prob_list.append(p)
#         merge_y_test = np.concatenate(y_test_list)
#         merge_y_prob = np.concatenate(y_prob_list)
#         # 计算roc_auc
#         roc_auc_all_mean = roc_auc_score(merge_y_test, merge_y_prob)

        
#         models_metrics.append({"model": model.__class__.__name__, "roc_auc_all_mean": float(roc_auc_all_mean), 
#                                "roc_aucs": roc_aucs, "accuracies": accuracies,
#                                "all_prob": all_prob,
#                                })


#         acc = np.array(accuracies)
#         roc_auc = np.array(roc_aucs)
#         print("model: ", model.__class__.__name__)
#         print("roc_auc mean: ", roc_auc.mean(), "roc_auc std: ", roc_auc.std(), "roc_auc max:", roc_auc.max())
#         roc_auc_list.append({'model': model.__class__.__name__, "roc_auc_max": roc_auc.max(), "roc_auc": list(roc_auc), "roc_auc_mean": roc_auc.mean()})
#         print("="*80)
#     return roc_auc_list, models_metrics


# def metrics_by_data_and_features(data, label, features, roc_auc_list):
#     kflod = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
#     #定义模型
#     models = [LogisticRegression(max_iter=MAX_ITER), KNeighborsClassifier(), SVC(probability=True), RandomForestClassifier(), MLPClassifier(hidden_layer_sizes=(5,), max_iter=500), xgboost.XGBClassifier()]
#     models_metrics = []
#     for model in models:
#         imputer = ColumnTransformer([
#                 ('knn_imputer', KNNImputer(n_neighbors=10), features),
#             ])
#         pipeline = Pipeline([
#                 ('imputer', imputer),
#                 ('classifier', model),
#             ])
#         # scores = cross_validate(pipeline, data.loc[:,top_features[:num_feature]], label, 
#         #                         cv=kflod, scoring=scoring)
        
#         # 存储每个样本的预测概率
#         probabilities = []
#         all_prob = {}
#         accuracies = []
#         roc_aucs = []
#         # 五折交叉验证
#         X = data.loc[:,features]
#         for train_index, test_index in kflod.split(X, label):
#             X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
#             y_train, y_test = label.iloc[train_index], label.iloc[test_index]

#             pipeline.fit(X_train, y_train)

#             # 预测概率
#             y_prob = pipeline.predict_proba(X_test)[:,1]
#             idx = list(y_test.index)
            
#             # id和概率对应
#             for id, prob in zip(idx, y_prob):
#                 all_prob[id] = float(prob)

#             probabilities.append((y_test.values, y_prob))

#             # 计算roc_auc
#             roc_auc = roc_auc_score(y_test, y_prob)
#             roc_aucs.append(roc_auc)

#             # 预测类别
#             y_pred = pipeline.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             accuracies.append(accuracy)
        
#         y_test_list = []
#         y_prob_list = []
#         for probability in probabilities:
#             t, p = probability
#             y_test_list.append(t)
#             y_prob_list.append(p)
#         merge_y_test = np.concatenate(y_test_list)
#         merge_y_prob = np.concatenate(y_prob_list)
#         # 计算roc_auc
#         roc_auc_all_mean = roc_auc_score(merge_y_test, merge_y_prob)

        
#         models_metrics.append({"model": model.__class__.__name__, "roc_auc_all_mean": float(roc_auc_all_mean), 
#                                "roc_aucs": roc_aucs, "accuracies": accuracies,
#                                "all_prob": all_prob,
#                                })


#         acc = np.array(accuracies)
#         roc_auc = np.array(roc_aucs)
#         print("model: ", model.__class__.__name__)
#         print("roc_auc mean: ", roc_auc.mean(), "roc_auc std: ", roc_auc.std(), "roc_auc max:", roc_auc.max())
#         roc_auc_list.append({'model': model.__class__.__name__, "roc_auc_max": roc_auc.max(), "roc_auc": list(roc_auc), "roc_auc_mean": roc_auc.mean()})
#         print("="*80)
#     return roc_auc_list, models_metrics

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
    start_time = time.time()
    # 临床+基因数据处理
    id_set = set(clinical_df.index) & set(raw_gene_df.index)
    raw_gene_filter = raw_gene_df.loc[list(id_set),:]
    raw_clinical_filter = clinical_df.loc[list(id_set),:]

    # 合并, 协变量也加上
    raw_gene_clinical = pd.concat([raw_gene_filter,raw_clinical_filter, co_df], axis=1)
    print("临床+基因数据shape: ", raw_gene_clinical.shape)

    gene_clinical = raw_gene_clinical.iloc[:,1:]
    gene_clinical_label = raw_gene_clinical["表型"]

    print("数据处理完毕")
    # 临床+基因模型
    print("临床+基因模型开始进行")

    # 数据归一化
    scaler = MinMaxScaler()
    gene_clinical_normalized = pd.DataFrame(scaler.fit_transform(gene_clinical), columns=gene_clinical.columns) 

    gene_clinical_sorted_dict, gene_clinical_top_features, gene_clinical_name_importance = \
        get_sorted_list_by_data(gene_clinical_normalized, gene_clinical_label)
    
    gene_clinical_num_feature = gene_clinical_sorted_dict[0]['features_numbers']

    

    # 保存结果目录
    import os
    save_path = f"co_result/normalized_211/{DROP_CLINICAL_RATE}"
    if args.debug:
        save_path = f"co_result/debug/{DROP_CLINICAL_RATE}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 保存临床+基因features和对应的roc_auc
    with open(os.path.join(save_path, "gene_clinical_features_roc_auc.json"), "w") as f:
        json.dump({"features_roc": gene_clinical_sorted_dict, 
                   "top_features": gene_clinical_top_features}, f)

    # 保存gene_clinical以及对应的shap值
    with open(os.path.join(save_path, "gene_clinical_name_importance.json"), "w") as f:
        json.dump(gene_clinical_name_importance, f)    

    # 临床+基因 各个模型概率和roc_auc的保存目录
    gene_clinical_roc_auc_list, gene_clinical_model_metrics = metrics_by_data_top_features(gene_clinical_normalized, 
                                                                                               gene_clinical_label, 
                                                                                                gene_clinical_top_features, 
                                                                                                gene_clinical_num_feature, 
                                                                                                roc_auc_list=[])
    # 保存临床+基因模型各个模型概率和roc_auc
    with open(os.path.join(save_path, "gene_clinical_model_metrics.json"), "w") as f:
        json.dump(gene_clinical_model_metrics, f)

    # 临床+基因模型中使用的特征
    gene_clinical_features = gene_clinical_top_features[:gene_clinical_num_feature]

    gene_clinical_len = len(gene_clinical_features)

    # 保存临床基因模型中使用的特征个数，以及特征
    gene_clinical_features_dict = {"gene_clinical_len": gene_clinical_len, "gene_clinical_features": list(gene_clinical_features)}

    
        
    # 临床+基因处理完毕，保存结果,用json保存
    import json
    with open(os.path.join(save_path, "gene_clinical_roc_auc_list.json"), "w") as f:
        json.dump(gene_clinical_roc_auc_list, f)
    
    print(f"gene_clinical_roc_auc_list save success, time: {(time.time() - start_time) / 60}min")

    start_time = time.time()
    
    # 拿到临床模型需要使用的特征
    clinical_features = []
    for feature in gene_clinical_features:
        if feature in clinical_df.columns:
            clinical_features.append(feature)
    # 临床特征加上协变量
    co_feature = "duration_of_diabetes"
    clinical_features.append(co_feature)

    # 拿到基因模型需要使用的特征
    gene_features = []
    for feature in gene_clinical_features:
        if feature in raw_gene_filter.columns:
            gene_features.append(feature)
    # 基因特征加上协变量
    gene_features.append(co_feature)

    print("特征提取完毕")

    clinical_len = len(clinical_features)
    gene_len = len(gene_features)

    # 临床数据归一化
    scaler = MinMaxScaler()
    # 临床数据与co_df合并
    clinical_df = pd.concat([clinical_df, co_df], axis=1)

    print("临床数据shape: ", clinical_df.shape)
    clinical_normalized = pd.DataFrame(scaler.fit_transform(clinical_df), columns=clinical_df.columns)

    # 临床模型
    clinical_roc_auc_list, clinical_model_metrics = metrics_by_data_and_features(clinical_normalized, clinical_label, clinical_features, roc_auc_list=[])
    
    # 临床处理完毕，保存结果,用json保存
    with open(os.path.join(save_path, "clinical_model_metrics.json"), "w") as f:
        json.dump(clinical_model_metrics, f)

    # 保存临床模型中使用的特征个数，以及特征
    clinical_features_dict = {"clinical_len": clinical_len, "clinical_features": list(clinical_features)}


    
    # 临床模型处理完毕，保存结果,用json保存
    with open(os.path.join(save_path, "clinical_roc_auc_list.json"), "w") as f:
        json.dump(clinical_roc_auc_list, f)
    print(f"clinical_roc_auc_list save success, time: {(time.time() - start_time) / 60}min")

    # 基因模型
    start_time = time.time()

    # 拿到数据和标签
    gene_df = raw_gene_df.iloc[:,1:]
    gene_label = raw_gene_df["表型"]

    # 基因数据归一化
    scaler = MinMaxScaler()

    # 基因数据与co_df合并
    gene_df = pd.concat([gene_df, co_df], axis=1)
    print("基因数据shape: ", gene_df.shape)

    gene_normalized = pd.DataFrame(scaler.fit_transform(gene_df), columns=gene_df.columns)

    gene_roc_auc_list, gene_model_metrics = metrics_by_data_and_features(gene_normalized, gene_label, gene_features, roc_auc_list=[])

    # 基因处理完毕，保存结果,用json保存
    with open(os.path.join(save_path, "gene_model_metrics.json"), "w") as f:
        json.dump(gene_model_metrics, f)

    # 保存基因模型中使用的特征个数，以及特征
    gene_features_dict = {"gene_len": gene_len, "gene_features": list(gene_features)}

    # 基因模型处理完毕，保存结果,用json保存
    with open(os.path.join(save_path, "gene_roc_auc_list.json"), "w") as f:
        json.dump(gene_roc_auc_list, f)
    print(f"gene_roc_auc_list save success, time: {(time.time() - start_time) / 60}min")

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




    

