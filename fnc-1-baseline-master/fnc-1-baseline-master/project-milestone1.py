import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
from fnc_kfold import generate_features, generate_features_without_labels
import pandas as pd


if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    unlabeled_competition_dataset = DataSet(name="competition_test", is_unlabeled=True)
    X_unlabeled = generate_features_without_labels(unlabeled_competition_dataset.stances, unlabeled_competition_dataset, "competition_unlabeled")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_unlabeled)]
    print(predicted)

    result = []
    for i, predictedLabel in enumerate(predicted):
        dict = {
            "Headline": unlabeled_competition_dataset.stances[i]['Headline'],
            "Body ID": unlabeled_competition_dataset.stances[i]['Body ID'],
            "Stance": predictedLabel
        }
        result.append(dict)

    test_data = pd.DataFrame(result)
    test_data.to_csv('answer.csv', index=False, encoding='utf-8')  # From pandas library

