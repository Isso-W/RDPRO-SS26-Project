import pandas as pd
import json
import os

#====================================
# step 2: basic dataset information
#====================================

def analyze_dataset(df, name):

    print(f"\n========== {name} Dataset ==========")

    # informations of rows and columns
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())


#====================================
# step 3: columns analysis
#====================================

def analyze_feature_type(series, column_name):

    col = column_name.lower()

    unique_count = series.nunique(dropna=True)

    total_count = len(series)

    # 1. Detect TARGET
    target_keywords = [
        "target",
        "label",
        "y",
        "survived",
        "income",
        "price",
        "saleprice"
    ]

    for keyword in target_keywords:
        if keyword in col:
            return "target"

    # 2. Detect ID
    if "id" in col:
        return "id"

    if (
        pd.api.types.is_integer_dtype(series)
        and unique_count == total_count
        and unique_count > 0.9 * total_count
    ):
        return "id"

    # 3. datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # 4. boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if unique_count == 2:
        return "categorical"

    # 5. numeric
    if pd.api.types.is_numeric_dtype(series):

        # ordinal detection
        if unique_count <= 10:

            values = series.dropna().unique()

            if len(values) > 0:

                sorted_vals = sorted(values)

                if all(
                    isinstance(v, (int, float))
                    for v in sorted_vals
                ):

                    if sorted_vals == list(
                        range(
                            int(min(sorted_vals)),
                            int(max(sorted_vals)) + 1
                        )
                    ):
                        return "ordinal"

            return "categorical"

        return "numeric"

    # 6. string types
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):

        avg_length = (
            series
            .dropna()
            .astype(str)
            .str.len()
            .mean()
        )

        # long text
        if avg_length > 30:
            return "text"

        # few categories
        if unique_count <= 50:
            return "categorical"

        return "text"

    return "unknown"
    

# columns analysis missing ratio
def analyze_missing_ratio(series):
    ratio = series.isnull().mean() * 100
    return float(round(ratio, 2))

# columns analysis unique values
def count_unique_values(series):
    return int(series.nunique())

# analyze columns of each dataset
def analyze_columns(df):

    column_info_list = []

    for col in df.columns:

        series = df[col]

        column_info = {}

        # column name
        column_info["name"] = col

        # pandas dtype
        column_info["dtype"] = str(series.dtype)

        # feature type
        column_info["feature_type"] = analyze_feature_type(series, col)

        # missing ratio
        column_info["missing_ratio"] = analyze_missing_ratio(series)

        # unique values
        column_info["unique_values"] = count_unique_values(series)

        # outlier count (for numeric)
        column_info["outlier_count"] = count_outliers(series)
        column_info_list.append(column_info)

    return column_info_list

# ====================================
# step 4: detect target variable
# ====================================

def detect_target(df):

    possible_targets = [
        "target",
        "label",
        "income",
        "survived",
        "price",
        "saleprice"
    ]

    for col in df.columns:

        col_lower = col.lower()

        for keyword in possible_targets:

            if keyword in col_lower:
                return col

    # fallback
    return df.columns[-1]

# =====================================
# step 5: detect task type
# =====================================
def detect_task_type(df, target_column):

    series = df[target_column]

    feature_type = analyze_feature_type(series, target_column)

    if feature_type == "numeric":

        unique_values = series.nunique()

        if unique_values <= 10:
            return "classification"

        return "regression"

    else:

        return "classification"
    
# =====================================
# step 6: detect outliers 
# =====================================
def count_outliers(series):

    if not pd.api.types.is_numeric_dtype(series):
        return 0

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    outliers = (
        (series < Q1 - 1.5 * IQR) |
        (series > Q3 + 1.5 * IQR)
    )

    return int(outliers.sum())

# ====================================
# generate json report
# ====================================
def generate_dataset_summary(
        df,
        dataset_name,
        columns_info,
        target_column,
        task_type
):

    dataset_summary = {}

    dataset_summary["dataset_name"] = dataset_name

    dataset_summary["n_rows"] = df.shape[0]

    dataset_summary["n_columns"] = df.shape[1]

    dataset_summary["target_column"] = target_column

    dataset_summary["task_type"] = task_type

    dataset_summary["columns"] = columns_info

    return dataset_summary

# json save function
def save_json(summary, filename):

    with open(filename, "w", encoding="utf-8") as f:

        json.dump(
            summary,
            f,
            indent=4
        )

def generate_summary_list(
        df,
        dataset_name,
        columns_info,
        target_column,
        task_type
):

    summary_list = []

    # Dataset info
    summary_list.append(
        f"Dataset {dataset_name} has "
        f"{df.shape[0]} rows and "
        f"{df.shape[1]} columns."
    )

    # Target info
    summary_list.append(
        f"Target column is {target_column}."
    )

    # Task type
    summary_list.append(
        f"Task type is {task_type}."
    )

    # Column summaries
    for col in columns_info:

        summary_list.append(
            f"Column {col['name']} "
            f"is {col['feature_type']} "
            f"with {col['missing_ratio']}% missing values "
            f"and {col['unique_values']} unique values."
        )

    return summary_list


def save_summary(summary_list, filename):

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    with open(filename, "w", encoding="utf-8") as f:

        for line in summary_list:

            f.write(line + "\n")

    print(f"Summary saved to {filename}")



# main function

def main():

    # step 1: load datasets

    titanic = pd.read_csv("data/titanic.csv")

    adult = pd.read_csv("data/adult.csv")

    house = pd.read_csv(
        "data/house_prices.csv",
        nrows = 10000
    )

    # step 2: basic info

    analyze_dataset(titanic, "Titanic")

    analyze_dataset(adult, "Adult")

    analyze_dataset(house, "House")

    # step 3: columns analysis

    # step 4: detect target variable

    target = detect_target(house)

    print("Target column:", target)

    # step 5: detect task type
    print("Titanic target:", detect_target(titanic))
    print("Titanic task:", detect_task_type(titanic, detect_target(titanic)))

    print("Adult target:", detect_target(adult))
    print("Adult task:", detect_task_type(adult, detect_target(adult)))

    print("House target:", detect_target(house))
    print("House task:", detect_task_type(house, detect_target(house)))
        
    # step 6: detect outliers
    print("\n===== Titanic Outlier Check =====")

    titanic_columns = analyze_columns(titanic)

    for col in titanic_columns:
        print(col)


    print("\n===== Adult Outlier Check =====")

    adult_columns = analyze_columns(adult)

    for col in adult_columns:
        print(col)


    print("\n===== House Outlier Check =====")

    house_columns = analyze_columns(house)

    for col in house_columns:
        print(col)

    # Step 7: save JSON report
    # Titanic JSON
    titanic_columns = analyze_columns(titanic)

    titanic_target = detect_target(titanic)

    titanic_task = detect_task_type(
        titanic,
        titanic_target
    )

    titanic_summary = generate_dataset_summary(
        titanic,
        "Titanic",
        titanic_columns,
        titanic_target,
        titanic_task
    )

    save_json(
        titanic_summary,
        "output/titanic_summary.json"
    )


    # Adult JSON
    adult_columns = analyze_columns(adult)

    adult_target = detect_target(adult)

    adult_task = detect_task_type(
        adult,
        adult_target
    )

    adult_summary = generate_dataset_summary(
        adult,
        "Adult",
        adult_columns,
        adult_target,
        adult_task
    )

    save_json(
        adult_summary,
        "output/adult_summary.json"
    )


    # House JSON
    house_columns = analyze_columns(house)

    house_target = detect_target(house)

    house_task = detect_task_type(
        house,
        house_target
    )

    house_summary = generate_dataset_summary(
        house,
        "House",
        house_columns,
        house_target,
        house_task
    )

    save_json(
        house_summary,
        "output/house_summary.json"
    )

    # generate summary text

    # Titanic summary
    titanic_summary_list = generate_summary_list(
        titanic,
        "Titanic",
        titanic_columns,
        titanic_target,
        titanic_task
    )

    save_summary(
        titanic_summary_list,
        "output/titanic_summary.txt"
    )


    # Adult summary
    adult_summary_list = generate_summary_list(
        adult,
        "Adult",
        adult_columns,
        adult_target,
        adult_task
    )

    save_summary(
        adult_summary_list,
        "output/adult_summary.txt"
    )


    # House summary
    house_summary_list = generate_summary_list(
        house,
        "House",
        house_columns,
        house_target,
        house_task
    )

    save_summary(
        house_summary_list,
        "output/house_summary.txt"
    )


# python entry point
if __name__ == "__main__":
    main()