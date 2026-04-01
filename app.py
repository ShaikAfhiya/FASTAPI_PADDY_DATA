from fastapi import FastAPI
import joblib
from pydantic import BaseModel,Field
import pandas as pd
#load model
model=joblib.load('paddy.pkl')
#fastapi
app=FastAPI()
class Data(BaseModel):
    hectares: float = Field(alias="Hectares")
    agriblock: str = Field(alias="Agriblock")
    variety: str = Field(alias="Variety")
    soil_types: str = Field(alias="Soil Types")
    seedrate_in_kg: float = Field(alias="Seedrate(in Kg)")
    lp_mainfield_in_tonnes: float = Field(alias="LP_Mainfield(in Tonnes)")
    nursery: str = Field(alias="Nursery")
    nursery_area_cents: float = Field(alias="Nursery area (Cents)")
    lp_nurseryarea_in_tonnes: float = Field(alias="LP_nurseryarea(in Tonnes)")
    dap_20days: float = Field(alias="DAP_20days")
    weed28d_thiobencarb: float = Field(alias="Weed28D_thiobencarb")
    urea_40days: float = Field(alias="Urea_40Days")
    potassh_50days: float = Field(alias="Potassh_50Days")
    micronutrients_70days: float = Field(alias="Micronutrients_70Days")
    pest_60day_in_ml: float = Field(alias="Pest_60Day(in ml)")
    rain_30d: float = Field(alias="30DRain( in mm)")
    ai_30d: float = Field(alias="30DAI(in mm)")
    rain_30_50d: float = Field(alias="30_50DRain( in mm)")
    ai_30_50d: float = Field(alias="30_50DAI(in mm)")
    rain_51_70d: float = Field(alias="51_70DRain(in mm)")
    ai_51_70d: float = Field(alias="51_70AI(in mm)")
    rain_71_105d: float = Field(alias="71_105DRain(in mm)")
    ai_71_105d: float = Field(alias="71_105DAI(in mm)")
    min_temp_d1_d30: float = Field(alias="Min temp_D1_D30")
    max_temp_d1_d30: float = Field(alias="Max temp_D1_D30")
    min_temp_d31_d60: float = Field(alias="Min temp_D31_D60")
    max_temp_d31_d60: float = Field(alias="Max temp_D31_D60")
    min_temp_d61_d90: float = Field(alias="Min temp_D61_D90")
    max_temp_d61_d90: float = Field(alias="Max temp_D61_D90")
    min_temp_d91_d120: float = Field(alias="Min temp_D91_D120")
    max_temp_d91_d120: float = Field(alias="Max temp_D91_D120")
    wind_speed_d1_d30: float = Field(alias="Inst Wind Speed_D1_D30(in Knots)")
    wind_speed_d31_d60: float = Field(alias="Inst Wind Speed_D31_D60(in Knots)")
    wind_speed_d61_d90: float = Field(alias="Inst Wind Speed_D61_D90(in Knots)")
    wind_speed_d91_d120: float = Field(alias="Inst Wind Speed_D91_D120(in Knots)")
    wind_direction_d1_d30: str = Field(alias="Wind Direction_D1_D30")
    wind_direction_d31_d60: str = Field(alias="Wind Direction_D31_D60")
    wind_direction_d61_d90: str = Field(alias="Wind Direction_D61_D90")
    wind_direction_d91_d120: str = Field(alias="Wind Direction_D91_D120")
    humidity_d1_d30: float = Field(alias="Relative Humidity_D1_D30")
    humidity_d31_d60: float = Field(alias="Relative Humidity_D31_D60")
    humidity_d61_d90: float = Field(alias="Relative Humidity_D61_D90")
    humidity_d91_d120: float = Field(alias="Relative Humidity_D91_D120")
    trash_in_bundles: float = Field(alias="Trash(in bundles)")
    class Config:
        allow_population_by_field_name = True
@app.post("/predict")
def predict(data: Data):
    try:
       df = pd.DataFrame([data.dict(by_alias=True)])
       prediction = model.predict(df)
       return {"prediction": prediction.tolist()}
    except Exception as e:
        return {'error':str(e)}
 





      
        
