from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
import pandas as pd

with open('../pickles/model', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open(r"../pickles/scaler", "rb") as input_file:
    scaler = pickle.load(input_file)

def pipeline(new):
    # new = pd.DataFrame(arr).T # single entry
    new.columns = ['neighbourhood_group', 'latitude', 'longitude', 'room_type',
       'minimum_nights', 'number_of_reviews', 'reviews_per_month',
       'calculated_host_listings_count', 'availability_365']
    
    for col in ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
       'calculated_host_listings_count', 'availability_365']:
        new[col]=new[col].astype(np.float)
        
    new['minimum_nights'] = np.log1p(new.loc[0,'minimum_nights'])

    new['all_year_avail'] = new['availability_365']>353
    new['low_avail'] = new['availability_365']< 12
    new['no_reviews'] = new['reviews_per_month']==0

    new['neighbourhood_group_City of Los Angeles'] = 1*(new['neighbourhood_group'] == 'City of Los Angeles')
    new['neighbourhood_group_Other Cities'] = 1*(new['neighbourhood_group'] == 'Other Cities')
    new['neighbourhood_group_Unincorporated Areas'] = 1*(new['neighbourhood_group'] == 'Unincorporated Areas')
    new['neighbourhood_group_unknown'] = 1*(new['neighbourhood_group'] == 'unknown')
    
    new['room_type_Entire home/apt'] = 1*(new['neighbourhood_group'] == 'Entire home/apt')
    new['room_type_Hotel room'] = 1*(new['neighbourhood_group'] == 'Hotel room')
    new['room_type_Private room'] = 1*(new['neighbourhood_group'] == 'Private room')
    new['room_type_Shared room'] = 1*(new['neighbourhood_group'] == 'Shared room')

    new.drop(['neighbourhood_group','room_type'], axis = 1, inplace = True)
    
    new_scaled = scaler.transform(new)
    return new_scaled
 
app = FastAPI(debug=True)

@app.get('/')
def home():
    return {'text': 'Los Angeles House Pricing Prediction'}

@app.get('/predict')
def predict(neighbourhood_group: str,
            latitude: float, 
            longitude: float, 
            room_type: str,
            minimum_nights: float, 
            number_of_reviews: float,
            reviews_per_month: float,
            calculated_host_listings_count: float,
            availability_365: float):
    
    arr = np.array([neighbourhood_group, latitude, longitude, room_type, 
                    minimum_nights, number_of_reviews, reviews_per_month,
                    calculated_host_listings_count, availability_365])
    new = pd.DataFrame(arr).T
    feature = pipeline(new)
    preds_scaled = model.predict(feature)
    preds = np.round(np.exp(preds_scaled[0]), 4)
    return {'The predicted price is {}'.format(preds)}

  
if __name__ == '__main__':
    uvicorn.run(app)
