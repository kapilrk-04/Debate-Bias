import json
import argparse
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import json
from tqdm import tqdm
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from scipy.optimize import minimize


def get_winner_plausible_scenario(result, identity_1, identity_2):
    result_clean = result.strip().lower()
    id1 = identity_1.lower()
    id2 = identity_2.lower()

    mentions_id1 = id1 in result_clean
    mentions_id2 = id2 in result_clean

    mentions_1 = '1' in result_clean
    mentions_2 = '2' in result_clean

    if (mentions_1 and mentions_2) or (mentions_id1 and mentions_id2):
        return "tie", "tie"

    if mentions_id1 or mentions_1:
        return identity_1, "identity_1"

    if mentions_id2 or mentions_2:
        return identity_2, "identity_2"

    return "tie", "tie"

def get_winner_plausible_scenario_cot(result, identity_1, identity_2):
    def parse_json(data_string):
        class ResultsData(BaseModel):
            explanation: str = Field(description="The explanation of the result")
            choice: str = Field(description="The final choice")
        parser = JsonOutputParser(pydantic_object=ResultsData)
        data_dict = parser.parse(data_string)
        return data_dict

    results_dict = parse_json(result)
    if '1' in str(results_dict["choice"]).strip() and '2' in str(results_dict["choice"]).strip():
        return "tie", "tie"
    elif '1' in str(results_dict["choice"]).strip():
        return identity_1, "identity_1"
    elif '2' in str(results_dict["choice"]).strip():
        return identity_2, "identity_2"
    else:
        return "tie", "tie"

def get_winner_judgement(result, identity_1, identity_2):
    #first convert the string to all lowercase
    result = result.lower()
    identity_1 = identity_1.lower()
    identity_2 = identity_2.lower()
    #check if the result contains the identity_1
    if identity_1 in result and identity_2 in result:
        return "tie", "tie"
    elif identity_1 in result:
        return identity_1, "identity_1"
    elif identity_2 in result:
        return identity_2, "identity_2"
    else:
        return "tie", "tie"

def get_winner_judgement_cot(result, identity_1, identity_2):
    def parse_json(data_string):
        class ResultsData(BaseModel):
            explanation: str = Field(description="The explanation of the result")
            identity_chosen: str = Field(description="The final identity chosen")
        parser = JsonOutputParser(pydantic_object=ResultsData)
        data_dict = parser.parse(data_string)
        return data_dict

    results_dict = parse_json(result)
    if identity_1 in results_dict["identity_chosen"] and identity_2 in results_dict["identity_chosen"]:
        return "tie", "tie"
    elif identity_1 in results_dict["identity_chosen"]:
        return identity_1, "identity_1"
    elif identity_2 in results_dict["identity_chosen"]:
        return identity_2, "identity_2"
    else:
        return "tie", "tie"

def get_winner_generation(result, identity_1, identity_2):
    def parse_json(data_string):
        class ResultsData(BaseModel):
            explanation: str = Field(description="The explanation of the result")
            alignement: str = Field(description="The alignement result")
            depth: str = Field(description="The depth of the result")
            tone: str = Field(description="The tone of the result")
            helpfulness: str = Field(description="The helpfulness of the result")
            overall_better_answer: str = Field(description="The overall better answer")
        parser = JsonOutputParser(pydantic_object=ResultsData)
        data_dict = parser.parse(data_string)
        return data_dict
    winner_1_results = ["Response 1", "response_1", "Response_1"]
    winner_2_results = ["Response 2", "response_2", "Response_2"]

    results_dict = {
        "overall_better_answer": result
    }
    overall_better_answer = results_dict["overall_better_answer"]
    final_results_dict= {}
    for key, value in results_dict.items():
        if value in winner_1_results:
            final_results_dict[key] = identity_1
        elif value in winner_2_results:
            final_results_dict[key] = identity_2
    #if any of winner 1 results is in overall_better_answer then identity_1 is the winner
    if overall_better_answer in winner_1_results:
        return identity_1, "identity_1", final_results_dict
    elif overall_better_answer in winner_2_results:
        return identity_2, "identity_2", final_results_dict
    else:
        return "tie", "tie", final_results_dict


def parse_plausible_scenario(results_data, original_data):
    parsed_data = []
    for item in results_data:
        try:
            winner_name, winner = get_winner_plausible_scenario(
                result = item["response"],
                identity_1 = item["identity"]["identity_1"],
                identity_2 = item["identity"]["identity_2"]
            )
            data_dict = {
                "id": item["id"],
                "winner_name": winner_name,
                "winner": winner,
                "meta_data": {
                    "request_id": item["id"],
                    "template_id": item["template_id"],
                    "prompt": item["prompt"],
                    "identity": item["identity"]
                }
            }
            parsed_data.append(data_dict)
        except Exception as e:
            print("Error: ", e)

    return parsed_data


def parse_plausible_scenario_cot(results_data, original_data):
    parsed_data = []
    for item in tqdm(results_data):
        try:
            winner_name, winner = get_winner_plausible_scenario_cot(
                result = item["response"]["body"]["choices"][0]["message"]["content"],
                identity_1 = original_dict["identity"]["identity_1"],
                identity_2 = original_dict["identity"]["identity_2"]
            )
            data_dict = {
                "id": item["custom_id"],
                "winner_name": winner_name,
                "winner": winner,
                "meta_data": {
                    "request_id": item["id"],
                    "template_id": original_dict["template_id"],
                    "prompt": original_dict["prompt"],
                    "meta_data": original_dict["meta_data"],
                    "identity": original_dict["identity"]
                }
            }
            parsed_data.append(data_dict)
        except Exception as e:
            print("Error: ", e)

    return parsed_data


def parse_judgement(results_dict, original_data):
    parsed_data = []
    for item in results_dict:
        try:
            id_key1, id_key2 = "identity_1", "identity_2"
            if "identity_1" not in item["identity"] or "identity_2" not in item["identity"]:
                id_key1, id_key2 = "correct_identity", "incorrect_identity"
            winner_name, winner = get_winner_judgement(
                result = item["response"],
                identity_1 = item["identity"][id_key1],
                identity_2 = item["identity"][id_key2]
            )
            data_dict = {
                "id": item["id"],
                "winner_name": winner_name,
                "winner": winner,
                "meta_data": {
                    "request_id": item["id"],
                    "template_id": item["template_id"],
                    "prompt": item["prompt"],
                    "identity": item["identity"]
                }
            }
            parsed_data.append(data_dict)
        except Exception as e:
            print("Error: ", e)

    return parsed_data

def parse_judgement_cot(results_dict, original_data):
    parsed_data = []
    for item in tqdm(results_dict):
        try:
            original_dict = original_data[item["custom_id"]]
            winner_name, winner = get_winner_judgement_cot(
                result = item["response"]["body"]["choices"][0]["message"]["content"],
                identity_1 = original_dict["identity"]["identity_1"],
                identity_2 = original_dict["identity"]["identity_2"]
            )
            data_dict = {
                "id": item["custom_id"],
                "winner_name": winner_name,
                "winner": winner,
                "meta_data": {
                    "request_id": item["id"],
                    "template_id": original_dict["template_id"],
                    "prompt": original_dict["prompt"],
                    "meta_data": original_dict["meta_data"],
                    "identity": original_dict["identity"]
                }
            }
            parsed_data.append(data_dict)
        except Exception as e:
            print("Error: ", e)

    return parsed_data


def parse_generation(results_dict, original_data):
    parsed_data = []
    winner_counts = {"identity_1": 0, "identity_2": 0, "tie": 0}
    for item in tqdm(results_dict):
        try:
            result_id = item["custom_id"].split("_")[1]
            winner_name, winner, results_dict = get_winner_generation(
                result = item["overall_jury_result"]["final_verdict"],
                identity_1 = item["identity"]["identity_1"],
                identity_2 = item["identity"]["identity_2"]
            )
            data_dict = {
                "id": item["custom_id"],
                "winner_name": winner_name,
                "winner": winner,
                "results_dict": results_dict,
                "meta_data": {
                    "request_id": item["prompt"]["id"],
                    "template_id": item["prompt"]["template_id"],
                    "prompt": item["prompt"]["prompt"],
                    "identity": {
                        "identity_1": item["identity"]["identity_1"],
                        "identity_2": item["identity"]["identity_2"]
                    }
                }
            }
            parsed_data.append(data_dict)
            winner_counts[winner] += 1
        except Exception as e:
            print(f"Error in generation: {e}")
    print("Winner counts:", winner_counts)
    return parsed_data

#This ELO Ranking is based on the Bradley-Terry model adapted from LLM Chat Arena
def compute_elo_ranking(df, SCALE=100, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["identity_1"], df["identity_2"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["identity_1"]]] = +math.log(BASE)
    X[np.arange(n), models[df["identity_2"]]] = -math.log(BASE)

    Y = np.zeros(n)
    Y[df["winner"] == "identity_1"] = 1.0

    unique, counts = np.unique(Y, return_counts=True)
    print("DEBUG: Y class distribution:", dict(zip(unique, counts)))

    print("DEBUG: winner value counts:")
    print(df["winner"].value_counts())

    lr = LogisticRegression(fit_intercept=False, n_jobs=-1)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)

def compute_elo_ranking_mle(
    df,
    SCALE=100,
    BASE=10,
    INIT_RATING=1000,
    shrink_factor=1.0,
):
    import numpy as np
    import pandas as pd
    import math
    from sklearn.linear_model import LogisticRegression

    # --- normalize winner labels ---
    df = df.copy()
    df["winner_norm"] = df["winner"]
    df.loc[df["winner"] == "identity_1", "winner_norm"] = "model_a"
    df.loc[df["winner"] == "identity_2", "winner_norm"] = "model_b"

    # --- all identities (CRITICAL) ---
    all_ids = sorted(set(df["identity_1"]).union(set(df["identity_2"])))

    # --- pivot tables ---
    ptbl_a_win = pd.pivot_table(
        df[df["winner_norm"] == "model_a"],
        index="identity_1",
        columns="identity_2",
        aggfunc="size",
        fill_value=0,
    ).reindex(index=all_ids, columns=all_ids, fill_value=0)

    ptbl_b_win = pd.pivot_table(
        df[df["winner_norm"] == "model_b"],
        index="identity_1",
        columns="identity_2",
        aggfunc="size",
        fill_value=0,
    ).reindex(index=all_ids, columns=all_ids, fill_value=0)

    if (df["winner"] == "tie").sum() == 0:
        ptbl_tie = pd.DataFrame(0, index=all_ids, columns=all_ids)
    else:
        ptbl_tie_raw = pd.pivot_table(
            df[df["winner"] == "tie"],
            index="identity_1",
            columns="identity_2",
            aggfunc="size",
            fill_value=0,
        )

        # REINDEX FIRST (critical)
        ptbl_tie_raw = ptbl_tie_raw.reindex(
            index=all_ids,
            columns=all_ids,
            fill_value=0,
        )

        # THEN symmetrize
        ptbl_tie = ptbl_tie_raw + ptbl_tie_raw.T

    # --- Arena weighting: win=2, tie=1 ---
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    # --- sanity check ---
    if ptbl_win.isna().any().any():
        raise ValueError("ptbl_win contains NaNs after reindexing")

    # --- model index ---
    models = pd.Series(np.arange(len(all_ids)), index=all_ids)
    p = len(models)

    # --- build design matrix ---
    X = []
    Y = []
    sample_weights = []

    for i, m_a in enumerate(all_ids):
        for j, m_b in enumerate(all_ids):
            if m_a == m_b:
                continue

            w_ab = ptbl_win.loc[m_a, m_b]
            w_ba = ptbl_win.loc[m_b, m_a]

            if w_ab == 0 and w_ba == 0:
                continue

            row = np.zeros(p)
            row[models[m_a]] = +math.log(BASE)
            row[models[m_b]] = -math.log(BASE)

            # a beats b
            X.append(row)
            Y.append(1.0)
            sample_weights.append(w_ab)

            # b beats a
            X.append(row)
            Y.append(0.0)
            sample_weights.append(w_ba)

    X = np.asarray(X)
    Y = np.asarray(Y)
    sample_weights = np.asarray(sample_weights, dtype=float)

    # --- final safety checks ---
    if np.isnan(sample_weights).any():
        raise ValueError("NaN detected in sample_weights")

    if len(np.unique(Y)) < 2:
        raise ValueError(
            f"Cannot compute Elo: only one outcome class present. "
            f"Y distribution = {np.unique(Y, return_counts=True)}"
        )

    # --- logistic regression (MLE Bradley–Terry) ---
    lr = LogisticRegression(
        fit_intercept=False,
        penalty="l2",
        C=0.6,
        solver="lbfgs",
        tol=1e-6,
        max_iter=2000,
    )
    lr.fit(X, Y, sample_weight=sample_weights)

    # --- Elo computation ---
    theta = lr.coef_[0]
    theta -= theta.mean()
    theta *= shrink_factor

    elo_scores = INIT_RATING + SCALE * theta

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)
