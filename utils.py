import xgboost
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from fancyimpute import KNN
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



MAX_ITER = 8000
def get_sorted_list_by_data(data, label):
    """输入数据和标签，得到roc_auc最好前n个特征"""
    model = LogisticRegression(max_iter=MAX_ITER)
    features = data.columns
    train_data = pd.DataFrame(KNN(k = 10).fit_transform(data), columns=features)
    model.fit(train_data, label)
    explainer = shap.LinearExplainer(model, train_data)

    shap_values = explainer(train_data)
    # 获取特征重要性
    feature_importances = np.abs(shap_values.values).mean(0)
    feature_names = train_data.columns

    # 获取特征名称和对应的重要性
    name_importance = {}
    for name, importance in zip(feature_names, feature_importances):
        name_importance[name] = importance

    # 排序
    sorted_idx = feature_importances.argsort()[::-1]
    top_features = [feature_names[i] for i in sorted_idx]
    # summarize the effects of all the features
    # 绘制特征重要性条形图并获取特征重要性数据
    # shap.summary_plot(shap_values, train_data, plot_type='bar', max_display=20)


    # 对前面的特征进行排序训练

    kflod = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scoring = ['accuracy','roc_auc']
    roc_auc_list = []
    for i in range(1, len(top_features) + 1):
        # xgb = xgboost.XGBClassifier()
        imputer = ColumnTransformer([
                ('knn_imputer', KNNImputer(n_neighbors=10), top_features[:i]),
            ])
        model = LogisticRegression(max_iter=MAX_ITER)
        pipeline = Pipeline([
                ('imputer', imputer),
                ('classifier', model),
            ])
        scores = cross_validate(pipeline, data.loc[:,top_features[:i]], label,
                                cv=kflod, scoring=scoring)
        acc = np.array(scores['test_accuracy'])
        roc_auc = np.array(scores['test_roc_auc'])
        print("features_numbers: ", i)
        # print("roc_auc mean: ", roc_auc.mean(), "roc_auc std: ", roc_auc.std(), "roc_auc max:", roc_auc.max())
        roc_auc_list.append({'features_numbers': i, "roc_auc_mean": roc_auc.mean(), "roc_auc": list(roc_auc)})
        # print("="*80)

    # 对roc_auc_list进行排序
    sorted_dict = sorted(roc_auc_list, key=lambda x:x['roc_auc_mean'], reverse=True)
    return sorted_dict, top_features, name_importance


def metrics_by_data_top_features(data, label, top_features, num_feature, roc_auc_list):
    """
    save_path: 保存预测概率的路径
    """
    kflod = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
    #定义模型
    models = [LogisticRegression(max_iter=MAX_ITER), KNeighborsClassifier(), SVC(probability=True), RandomForestClassifier(), MLPClassifier(hidden_layer_sizes=(5,), max_iter=500), xgboost.XGBClassifier()]
    models_metrics = []
    for model in models:
        imputer = ColumnTransformer([
                ('knn_imputer', KNNImputer(n_neighbors=10), top_features[:num_feature]),
            ])
        pipeline = Pipeline([
                ('imputer', imputer),
                ('classifier', model),
            ])
        # scores = cross_validate(pipeline, data.loc[:,top_features[:num_feature]], label, 
        #                         cv=kflod, scoring=scoring)
        
        # 存储每个样本的预测概率
        probabilities = []
        all_prob = {}
        accuracies = []
        roc_aucs = []
        # 五折交叉验证
        X = data.loc[:,top_features[:num_feature]]
        for train_index, test_index in kflod.split(X, label):
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = label.iloc[train_index], label.iloc[test_index]

            pipeline.fit(X_train, y_train)

            # 预测概率
            y_prob = pipeline.predict_proba(X_test)[:,1]
            idx = list(y_test.index)
            
            # id和概率对应
            for id, prob in zip(idx, y_prob):
                all_prob[id] = float(prob)

            probabilities.append((y_test.values, y_prob))

            # 计算roc_auc
            roc_auc = roc_auc_score(y_test, y_prob)
            roc_aucs.append(roc_auc)

            # 预测类别
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        y_test_list = []
        y_prob_list = []
        for probability in probabilities:
            t, p = probability
            y_test_list.append(t)
            y_prob_list.append(p)
        merge_y_test = np.concatenate(y_test_list)
        merge_y_prob = np.concatenate(y_prob_list)
        # 计算roc_auc
        roc_auc_all_mean = roc_auc_score(merge_y_test, merge_y_prob)

        
        models_metrics.append({"model": model.__class__.__name__, "roc_auc_all_mean": float(roc_auc_all_mean), 
                               "roc_aucs": roc_aucs, "accuracies": accuracies,
                               "all_prob": all_prob,
                               })


        acc = np.array(accuracies)
        roc_auc = np.array(roc_aucs)
        print("model: ", model.__class__.__name__)
        print("roc_auc mean: ", roc_auc.mean(), "roc_auc std: ", roc_auc.std(), "roc_auc max:", roc_auc.max())
        roc_auc_list.append({'model': model.__class__.__name__, "roc_auc_max": roc_auc.max(), "roc_auc": list(roc_auc), "roc_auc_mean": roc_auc.mean()})
        print("="*80)
    return roc_auc_list, models_metrics


def metrics_by_data_and_features(data, label, features, roc_auc_list):
    kflod = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
    #定义模型
    models = [LogisticRegression(max_iter=MAX_ITER), KNeighborsClassifier(), SVC(probability=True), RandomForestClassifier(), MLPClassifier(hidden_layer_sizes=(5,), max_iter=500), xgboost.XGBClassifier()]
    models_metrics = []
    for model in models:
        imputer = ColumnTransformer([
                ('knn_imputer', KNNImputer(n_neighbors=10), features),
            ])
        pipeline = Pipeline([
                ('imputer', imputer),
                ('classifier', model),
            ])
        # scores = cross_validate(pipeline, data.loc[:,top_features[:num_feature]], label, 
        #                         cv=kflod, scoring=scoring)
        
        # 存储每个样本的预测概率
        probabilities = []
        all_prob = {}
        accuracies = []
        roc_aucs = []
        # 五折交叉验证
        X = data.loc[:,features]
        for train_index, test_index in kflod.split(X, label):
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = label.iloc[train_index], label.iloc[test_index]

            pipeline.fit(X_train, y_train)

            # 预测概率
            y_prob = pipeline.predict_proba(X_test)[:,1]
            idx = list(y_test.index)
            
            # id和概率对应
            for id, prob in zip(idx, y_prob):
                all_prob[id] = float(prob)

            probabilities.append((y_test.values, y_prob))

            # 计算roc_auc
            roc_auc = roc_auc_score(y_test, y_prob)
            roc_aucs.append(roc_auc)

            # 预测类别
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        y_test_list = []
        y_prob_list = []
        for probability in probabilities:
            t, p = probability
            y_test_list.append(t)
            y_prob_list.append(p)
        merge_y_test = np.concatenate(y_test_list)
        merge_y_prob = np.concatenate(y_prob_list)
        # 计算roc_auc
        roc_auc_all_mean = roc_auc_score(merge_y_test, merge_y_prob)

        
        models_metrics.append({"model": model.__class__.__name__, "roc_auc_all_mean": float(roc_auc_all_mean), 
                               "roc_aucs": roc_aucs, "accuracies": accuracies,
                               "all_prob": all_prob,
                               })


        acc = np.array(accuracies)
        roc_auc = np.array(roc_aucs)
        print("model: ", model.__class__.__name__)
        print("roc_auc mean: ", roc_auc.mean(), "roc_auc std: ", roc_auc.std(), "roc_auc max:", roc_auc.max())
        roc_auc_list.append({'model': model.__class__.__name__, "roc_auc_max": roc_auc.max(), "roc_auc": list(roc_auc), "roc_auc_mean": roc_auc.mean()})
        print("="*80)
    return roc_auc_list, models_metrics