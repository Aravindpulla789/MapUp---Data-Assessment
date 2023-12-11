#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd


# In[12]:


import numpy as np


# In[13]:


def generate_car_matrix(dataset):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset, index_col="id_1")

    # Pivot the DataFrame to create the desired matrix
    result_matrix = df.pivot(columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    result_matrix.values[[range(len(result_matrix))]*2] = 0

    return result_matrix


# In[14]:


dataset_path ="C:\\Users\\ARAVIND PULLA\\Downloads\\dataset-1.csv"
result_df = generate_car_matrix(dataset_path)
print(result_df)


# In[15]:


def get_type_count(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Add a new categorical column 'car_type' based on the 'car' column values
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], include_lowest=True, right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts


# In[16]:


file_path ="C:\\Users\\ARAVIND PULLA\\Downloads\\dataset-1.csv"
result = get_type_count(file_path)
print(result)


# In[17]:


def get_bus_indexes(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean value
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    sorted_bus_indexes = sorted(bus_indexes)

    return sorted_bus_indexes


# In[18]:


file_path = "C:\\Users\\ARAVIND PULLA\\Downloads\\dataset-1.csv"
result = get_bus_indexes(file_path)
print(result)


# In[19]:


def filter_routes(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Group by 'route' and calculate the average of the 'truck' column for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' column is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    sorted_selected_routes = sorted(selected_routes)

    return sorted_selected_routes


# In[20]:


file_path = "C:\\Users\\ARAVIND PULLA\\Downloads\\dataset-1.csv"
result = filter_routes(file_path)
print(result)


# In[21]:


def multiply_matrix(result_df):
    # Copy the DataFrame to avoid modifying the original DataFrame
    modified_df = result_df.copy()

    # Apply the specified logic to modify values in the DataFrame
    modified_df = modified_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df


# In[22]:


modified_result_df = multiply_matrix(result_df)
print(modified_result_df)


# In[23]:


# question 6
def verify_time_completeness(data):
    # Convert 'timestamp' to datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Extract day and time information
    data['day'] = data['timestamp'].dt.day_name()
    data['hour'] = data['timestamp'].dt.hour

    # Define a function to check if a pair has incorrect timestamps
    def check_timestamps(group):
        # Check if all days of the week are present
        days_present = set(group['day'])
        all_days_present = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']) <= days_present

        # Check if the pair covers a full 24-hour period
        full_day_coverage = set(group['hour']) == set(range(24))

        return not (all_days_present and full_day_coverage)

    # Apply the function to each group (id, id_2)
    result = data.groupby(['id', 'id_2']).apply(check_timestamps)

    return result


# In[24]:


file_path ="C:\\Users\\ARAVIND PULLA\\Downloads\\dataset-2.csv"
result_series = verify_time_completeness(file_path)
print(result_series)


# In[13]:


##question 6
from datetime import datetime, timedelta

def check_timestamp_completeness(df):
    # Combine 'startDay' and 'startTime' to create a datetime column 'start_datetime'
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

    # Combine 'endDay' and 'endTime' to create a datetime column 'end_datetime'
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Check if the timestamps cover a full 24-hour period
    full_24_hours = (df['end_datetime'] - df['start_datetime']) == timedelta(days=1)

    # Check if the timestamps span all 7 days of the week
    span_all_days = df.groupby(['id', 'id_2'])['start_datetime'].apply(
        lambda x: x.dt.dayofweek.nunique() == 7
    )

    # Combine the two conditions using the bitwise AND operator
    completeness_check = full_24_hours & span_all_days

    return completeness_check


# In[2]:


file_path ="C:\\Users\\ARAVIND PULLA\\Downloads\\dataset-2.csv"
result_series = verify_time_completeness(df)
print(result_series)


# In[15]:


file_path ="C:\\Users\\ARAVIND PULLA\\Downloads\\dataset-2.csv"
df = pd.read_csv(file_path)

# Set multi-index (id, id_2) for the result
df.set_index(['id', 'id_2'], inplace=True)

completeness_result = check_timestamp_completeness(df)
print(completeness_result)


# In[22]:


## python test 2


# In[20]:


def calculate_distance_matrix(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Create a pivot table to represent distances between toll locations
    distance_matrix = df.pivot(index='id_start', columns='id_end', values='distance')

    # Fill missing values by transposing and adding with the original matrix
    distance_matrix = distance_matrix.fillna(0) + distance_matrix.fillna(0).T

    # Fill diagonal values with 0
    distance_matrix.values[[range(len(distance_matrix))]*2] = 0

    return distance_matrix


# In[21]:


file_path ="C:\\Users\\ARAVIND PULLA\\Downloads\\dataset-3.csv"
distance_matrix = calculate_distance_matrix(file_path)
print(distance_matrix)


# In[23]:


def unroll_distance_matrix(distance_matrix):
    # Extract the lower triangle of the distance matrix excluding the diagonal
    lower_triangle = distance_matrix.where(np.tril(np.ones(distance_matrix.shape), k=-1).astype(bool))

    # Find the indices of non-zero values in the lower triangle
    indices = np.where(lower_triangle.notna())

    # Create a DataFrame with id_start, id_end, and distance
    unrolled_df = pd.DataFrame({
        'id_start': distance_matrix.index[indices[0]],
        'id_end': distance_matrix.columns[indices[1]],
        'distance': lower_triangle.values[indices]
    })

    return unrolled_df


# In[25]:


unrolled_distance_df = unroll_distance_matrix(distance_matrix)
print(unrolled_distance_df)


# In[26]:


def find_ids_within_ten_percentage_threshold(distance_df, reference_value):
    # Filter DataFrame for rows where id_start is equal to the reference value
    reference_rows = distance_df[distance_df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_average = reference_rows['distance'].mean()

    # Calculate the lower and upper bounds for the 10% threshold
    lower_bound = reference_average - 0.1 * reference_average
    upper_bound = reference_average + 0.1 * reference_average

    # Filter DataFrame for rows within the 10% threshold
    within_threshold_df = distance_df[
        (distance_df['id_start'] != reference_value) &  # Exclude the reference value itself
        (distance_df['distance'] >= lower_bound) &
        (distance_df['distance'] <= upper_bound)
    ]

    # Get the unique values from the id_start column and sort them
    within_threshold_ids = sorted(within_threshold_df['id_start'].unique())

    return within_threshold_ids


# In[27]:


reference_value = 1  # Replace with the desired reference value
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_df, reference_value)
print(result_ids_within_threshold)


# In[28]:


def calculate_toll_rate(distance_df):
    # Copy the input DataFrame to avoid modifying the original
    df_with_toll_rates = distance_df.copy()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df_with_toll_rates[vehicle_type] = df_with_toll_rates['distance'] * rate_coefficient

    return df_with_toll_rates


# In[29]:


distance_df_with_toll = calculate_toll_rate(unrolled_distance_df)
print(distance_df_with_toll)


# In[30]:


from datetime import datetime, time, timedelta

def calculate_time_based_toll_rates(distance_df):
    # Copy the input DataFrame to avoid modifying the original
    df_with_time_based_toll = distance_df.copy()

    # Define time ranges for weekdays and weekends
    weekday_time_ranges = [
        (time(0, 0, 0), time(10, 0, 0)),
        (time(10, 0, 0), time(18, 0, 0)),
        (time(18, 0, 0), time(23, 59, 59))
    ]
    
    weekend_time_range = (time(0, 0, 0), time(23, 59, 59))

    # Define discount factors for each time range
    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Generate time-based toll rates for each time range
    for day in range(7):  # 0 represents Monday, 1 represents Tuesday, and so on
        for start_time, end_time in weekday_time_ranges:
            mask = (df_with_time_based_toll['start_day'] == day) &                    (df_with_time_based_toll['end_day'] == day) &                    (df_with_time_based_toll['start_time'] >= start_time) &                    (df_with_time_based_toll['end_time'] <= end_time)
            df_with_time_based_toll.loc[mask, 'time_based_toll'] *= weekday_discount_factors[weekday_time_ranges.index((start_time, end_time))]

        # Apply weekend discount factor
        mask = (df_with_time_based_toll['start_day'] == day) &                (df_with_time_based_toll['end_day'] == day)
        df_with_time_based_toll.loc[mask, 'time_based_toll'] *= weekend_discount_factor

    # Convert time columns to datetime.time objects
    df_with_time_based_toll['start_time'] = df_with_time_based_toll['start_time'].apply(lambda x: time(*map(int, str(x).split(':'))))
    df_with_time_based_toll['end_time'] = df_with_time_based_toll['end_time'].apply(lambda x: time(*map(int, str(x).split(':'))))

    return df_with_time_based_toll


# In[31]:


distance_df_with_time_based_toll = calculate_time_based_toll_rates(distance_df_with_toll)
print(distance_df_with_time_based_toll)


# In[ ]:




