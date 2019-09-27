"""This File contains functions for retrieving data from our postgresql server and cleaning it"""
import psycopg2 as pg
import pandas as pd
from sklearn.model_selection import train_test_split


def download_data(key):
    """This function takes as an argument the server password as a string and returns
    a dataframe """
    #"K*%4t3VK0ab%gn"
    
    # this is a list of all the columns in the busniness database, we only want to grab
    # the ones that are not commented out, for various reasons.
    columns_business = [
        'business_id',
        # 'name',
        # 'address',
        #'city',
        'state',
        #'postal_code',
        'latitude',
        'longitude',
        # 'stars',
        'review_count',
        'attributes_goodforkids',
        #'categories',   further processing
        'is_open',
        # 'hours_monday',
        # 'hours_tuesday',
        # 'hours_wednesday',
        # 'hours_thursday',
        # 'hours_friday',
        # 'hours_saturday',
        # 'hours_sunday',
        'attributes_restaurantsreservations',
        # 'attributes_goodformeal',
        # 'attributes_businessparking',
        'attributes_caters',
        'attributes_noiselevel',
        'attributes_restaurantstableservice',
        'attributes_restaurantstakeout',
        'attributes_restaurantspricerange2',
        'attributes_outdoorseating',
        'attributes_bikeparking',
        # 'attributes_ambience',
        'attributes_hastv',
        'attributes_wifi',
        'attributes_alcohol',
        'attributes_restaurantsattire',
        'attributes_restaurantsgoodforgroups',
        'attributes_restaurantsdelivery',
        'attributes_businessacceptscreditcards',
        'attributes_businessacceptsbitcoin',
        #'attributes_byappointmentonly',
        #'attributes_acceptsinsurance',
        # 'attributes_music',
        'attributes_goodfordancing',
        'attributes_coatcheck',
        'attributes_happyhour',
        # 'attributes_bestnights',
        'attributes_wheelchairaccessible',
        'attributes_dogsallowed',
        # 'attributes_byobcorkage',
        'attributes_drivethru',
        'attributes_smoking',
        #'attributes_agesallowed',
        #'attributes_hairspecializesin',
        #'attributes_corkage',
        #'attributes_byob',
        #'attributes_dietaryrestrictions', further processing
        #'attributes_open24hours',
        #'attributes_restaurantscounterservice',
        'restaurant',
        'meanfunny',
        'meanuseful',
        'avgwordcount',
        'maxwordcount',
        'minwordcount',
        'avgfunnywordcount',
        'maxfunnywordcount',
        'avgusefulwordcount',
        'maxusefulwordcount',
        'medianwordcount',
        'upperquartilewordcount',
        'lowerquartilewordcount',
        'target']
    
    #this sets up the query we will make to the data base, the upper and lower number of review cut off
    #points, and the columns we want
    list_as_string =', '.join(columns_business)
    u_cutoff=87
    l_cutoff=10
    
    # this connects to the database, gets our data into a dataframe and closes the connection
    con = pg.connect(database="postgres", user="flatiron_user_1", password=key, host="34.74.239.44", port="5432")
    cur = con.cursor()
    
    cur.execute(f"SELECT {list_as_string} FROM business WHERE restaurant IS true AND REVIEW_COUNT < {u_cutoff} AND REVIEW_COUNT > {l_cutoff}")
    business_data=cur.fetchall()
    business_data=pd.DataFrame(business_data)
    business_data.columns=columns_business
    return business_data

def clean_data(data):
    """this function takes a dataframe as an argument, cleans it, and returns the cleaned
    and data."""
    
    cols = data.columns
    
    #these columns had some extra characters in the strings becuase of encoding issues
    list_to_strip=[
    'attributes_alcohol',
    'attributes_restaurantsattire',
    'attributes_wifi',
    'attributes_smoking',
    'attributes_noiselevel',
              ]
    #this removes quotation marks and u's from strings
    
    for col in list_to_strip:
        data[col]=data[col].str.strip("u\'")
    
    #this replaces the strings None and none with Nan objects
    for col in cols:
        data[col]=data[col].where(data[col]!='None')
        data[col]=data[col].where(data[col]!='none')
    
    #this creates a list of categorical and numerical features
    categorical_features = cols.drop([
        'review_count',
        'restaurant',
        'latitude',
        'longitude',
        'business_id',
        'meanfunny',
        'meanuseful',
        'avgwordcount',
        'maxwordcount',
        'minwordcount',
        'avgfunnywordcount',
        'maxfunnywordcount',
        'avgusefulwordcount',
        'maxusefulwordcount',
        'medianwordcount',
        'upperquartilewordcount',
        'lowerquartilewordcount',
        'target'])
    
                
    numerical_features = [
        'review_count',
        'latitude',
        'longitude',
        'meanfunny',
        'meanuseful',
        'avgwordcount',
        'maxwordcount',
        'minwordcount',
        'avgfunnywordcount',
        'maxfunnywordcount',
        'avgusefulwordcount',
        'maxusefulwordcount',
        'medianwordcount',
        'upperquartilewordcount',
        'lowerquartilewordcount']
    
    #this replaces the categorial nans with 9 as a placeholder and fills numerical nans with 0
    data[categorical_features]=data[categorical_features].fillna(9)
    data[numerical_features]=data[numerical_features].fillna(0)
    
    #this makes all the categorical columns strings
    data[categorical_features]=data[categorical_features].astype(str)
    data = data
    
    return data, numerical_features, categorical_features

def tt_split(dataframe,numerical_features, categorical_features):
    data=dataframe.drop(['business_id','restaurant','target'],axis=1)
    data=pd.concat([data[numerical_features],data[categorical_features]],axis=1)
    target=dataframe.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=50)
    return X_train, X_test, y_train, y_test