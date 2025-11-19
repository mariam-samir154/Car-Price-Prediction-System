import streamlit as st
import joblib
import pandas as pd
import pickle
selected_features = ['Airbags', 'Prod. year', 'Mileage', 'Drive wheels', 'Wheel',
                     'Gear box type', 'Manufacturer', 'Model', 'Levy', 'Fuel type',
                     'Leather interior', 'Category', 'Cylinders']

model = joblib.load('linear_regression_model_new_1.pkl')
scaler = joblib.load('scaler_l.pkl')

svr_model = joblib.load("svr_model.pkl")
svr_scaler = joblib.load("svr_scaler.pkl")

# neural_model = joblib.load("neural_model.pkl")
# neural_scaler = joblib.load("neural_scaler.pkl")

xgb_model = joblib.load("xgb_model.pkl")
xgb_scaler = joblib.load("xgb_scaler.pkl")

rf_model = joblib.load("rf_model.pkl")
rf_scaler = joblib.load("rf_scaler.pkl")

with open("svr_metrics.pkl", "rb") as f:
    svr_metrics = joblib.load(f)
with open("rf_metrics.pkl", "rb") as f:
    rf_metrics = joblib.load(f)
with open("xgb_metrics.pkl", "rb") as f:
    xgb_metrics = joblib.load(f)
# with open("neural_metrics.pkl", "rb") as f:
#     neural_metrics = joblib.load(f)


with open('encoders/category_encoder_1.pkl', 'rb') as f:
    category_encoder = joblib.load(f)
with open('encoders/color_encoder_1.pkl', 'rb') as f:
    color_encoder = joblib.load(f)
with open('encoders/doors_encoder_1.pkl', 'rb') as f:
    doors_encoder = joblib.load(f)
with open('encoders/drive_encoder.pkl', 'rb') as f:
    drive_encoder = joblib.load(f)
with open('encoders/fuel_encoder_1.pkl', 'rb') as f:
    fuel_encoder = joblib.load(f)
with open('encoders/gear_encoder_1.pkl', 'rb') as f:
    gear_encoder = joblib.load(f)
with open('encoders/leather_encoder_1.pkl', 'rb') as f:
    leather_encoder = joblib.load(f)
with open('encoders/manufacturer_encoder_1.pkl', 'rb') as f:
    manufacturer_encoder = joblib.load(f)
with open('encoders/model_encoder_1.pkl', 'rb') as f:
    model_encoder = joblib.load(f)
with open('encoders/wheel_encoder_1.pkl', 'rb') as f:
    wheel_encoder = joblib.load(f)

st.title("Car Price Prediction App")
model_option = st.selectbox("Choose a model", ["Linear Regression", "SVR","Random Forest","XGBoost"])



test_r2=0
if model_option == "Linear Regression":
    pred_model = model
    pred_scaler = scaler
    pred_metrics = svr_metrics
    test_r2 = 0.80
elif model_option == "SVR":
    pred_model = svr_model
    pred_scaler = svr_scaler
    pred_metrics = svr_metrics
    test_r2 = 0.663
# elif model_option == "Neural Network":
#     pred_model = neural_model
#     pred_scaler = neural_scaler
#     pred_metrics = neural_metrics
#     test_r2 = 0.7345
elif model_option == "XGBoost":
    pred_model = xgb_model
    pred_scaler = xgb_scaler
    pred_metrics = xgb_metrics
    test_r2 = 0.7345
elif model_option == "Random Forest":
    pred_model = rf_model
    pred_scaler = rf_scaler
    pred_metrics = rf_metrics
    test_r2 = 0.763



drive_wheels_options = ['Front', '4x4', 'Rear']
wheel_options = ['Left wheel', 'Right-hand drive']
gear_box_options = ['Automatic', 'Variator' ,'Manual', 'Tiptronic']
air_bag_options=[12.0, 4.0, 0.0, 8.0, 7.0, 10.0, 2.0, 6.0, 16.0, 1.0, 3.0, 9.0, 11.0, 5.0, 13.0, 15.0, 14.0]
model_options=['CLA 250', 'Actyon', 'ML 350', 'H1', 'Genesis', 'Sonata', 'RX 450', 'E 500', 'Santa FE', '320 Diesel', 'C 280', 'Astra', 'Sportage', 'XC90 3.2 AWD', 'Tiida', 'Corolla', '4Runner', 'Land Cruiser 105', 'CLK 230', 'Prius', 'Corolla LE', 'CHR', '330', 'S 350 CDI 320', 'Camry', 'FIT', 'Fiesta', 'Highlander', '528', 'LS 460', 'M5 Japan', '500', 'Aqua', 'GL 320', 'Cr-v LX', 'C 220', 'GX 460', 'Prius V', 'Passat', 'GX 470', '320', 'Prius C', 'CX-5', 'Malibu', 'X5', 'Q7', 'Tucson', 'Fusion', 'Grand Vitara', 'A7', 'E 350', 'Elantra', 'Element', 'Delica', 'Civic', 'Altima', 'Vectra', 'Forester', 
'Sorento', 'Jetta', 'CLK 320', 'E 290', 'Sprinter 308 CDI', 'E 200', 'Vanette', 'C-MAX', 'RAV 4', 'Escape', 'MKZ', 'Sonata SPORT', 'Note', 'C 200 KOMPRESSOR', 'F150', 'GLA 250', 'Elantra LIMITEDI', '31514', '500L', 'A4', 'Captiva', 'Equinox', '525 Vanos', 'Camry XV50', 'Avenger', '328', 'Kyron', 'Century', 'E 320', 'Compass', 'Cruze', '2107', 'Shuttle', 'Tundra', 'Gentra', 'X-Trail', '500C Lounge', 'Legacy Outback', 'Cayenne', 'Sienna', 'Hilux', 'CT 200h', 'E 300', 'Spark', 'Optima', 'Vito', 'Juke', 'E 220', 'Terios', 'Malibu LT', 'Prius V HIBRID', 'Odyssey', 'Lacetti', 'Transit', 'RX 350', 'Combo', 'RAM', 'Avalon LIMITED', 'C 180', 'Q5', 'Monterey', 'X6', 'Camaro', 'Orlando', '21', 'Elantra se', 'Corolla se', 'Step Wagon', 'VOXY', 'REXTON', '325', 'X-Terra', 'E 240', 'A6', 'CLK 200', 'E 550', 'Tacoma', 'Caliber', 'Explorer Turbo japan', 'Colt', 'S 320', 'Focus', 'Astra H', 'Corsa', 'NX 300', 'E-pace', 'Mazda 6', 'Murano', 'Wrangler', '318', 'Pajero IO', 'Montero Sport', 'Insight', 'Golf', 'Vitz', 'Volt', 'Challenger', 'CX-7', 'Land Cruiser Prado', 'Impala', '535', 'C1 C', 'Outlander', '2109', 'Frontera', '24', 'A 190', 'S 550', 'Land Cruiser Prado RX', 'Juke Nismo', 'Elantra Limited', 'Sprinter', 'Camry Le', 'Renegade', 'Pajero 2', 'Avensis', 'CLS 55 AMG', 'Cr-v', 'Yaris', 'C 220 CDI', 'Galant GTS', 'Cruze LT RS', 'Avella', 'SOUL', '530', 'RAV 4 LIMITED', 'Maxima', 'GLE 350', 'CC', 'SLK 32 AMG', 'Versa', '1000', 'ML 63 AMG', 'VehiCross', 'B 180', 'One', 'Quest', 'Transit Connect Prastoi', 'Pajero', 'RIO', 'Astra g', 'Lantra LIMITED', 'Mustang', 'Sprinter VAN', 'Golf 4', 'Highlander XLE', 'Aveo', 'Korando', 'Mx-5', 'IS 200', 'Vento', 'Camry se', 'March', 'Camry LE', 'C 300', 'A8', 'Aqua s', 'C 36 AMG', 'CL 600', 'Matiz', 'C 250', '200', 'RX 400', 
'S 600', '320 i', '520', 'Swift', 'HUSTLER', 'CL550 AMG', '750 4.8', 'FIT RS', 'Scirocco', 'Alphard', 'Intrepid', 'Cr-v Cr-v', 'Camry SE', 'IS 250', 'E 420', 'S 63 AMG', 'INSIGNIA', 'Zafira', 'Kizashi', 'R 350', 'E 270', 'RX 400 hybrid', 'Mazda 2', 'Omega B', 'Airtrek', 'TERRAIN', 'Combo TDI', 'A 140', 'Accent', 'G 63 AMG', 'Sentra', 'Range Rover', 'C 350', 'Sequoia', '328 Xdrive', 'Cerato', 'Avalon Limited', 'C 200', 'Doblo', '535 M PAKET', '740', 'Fun Cargo', 'X-Trail NISMO', 'CLS 500', 'XF', 'A3', 'SX4', 'Cruze LT', 'GL 63 AMG', 'C 320', 'Altezza', 'DS 4', 'ML 320', 'Ghibli', 'ATS', 'Passat se', 'FIT GP-6', '1500', 'X5 E70', 'XV', 'Ramcharger', '370Z', 'Juke NISMO', 'NX 200', '750', '545', 'A6 evropuli', 'Sirion', 'Skyline', 
'macan', 'Verso', 'Liberty', 'Cooper', 'Elysion', '328 DIZEL', 'Accent SE', '307', 'Taurus', 'Ist 1.5', '528 i', 'GL 500', 
'Edix FR-v', 'Phaeton', 'CX-9', '320 Gran Turismo', 'Outback', '535 I', 'Land Cruiser 200', 'Legacy Bl5', 'Laguna', '3008 2.0', 'iA isti', 'Rodeo', 'I30', '435 CUPE', 'ML 350 4matic', 'X5 3.5', 'Transit Tourneo', 'Camry Hybrid', '316', 'Transit Connect', 'Scenic', 'M3', 'Highlander LIMITED', 'ES 350', 'Dart', 'ES 300', 'Gloria', 'Legacy', 'Serena', 'Pilot', 'CLA 250 AMG', 'Encore', 'Cayenne S', '911', 'G 65 AMG ', 'GL 350 BLUETEC', 'L 200', 'UP', '190', 'CHR Limited', 'Smart', 'S 420', 'ML 550', 'C8', 'Jimny', 'Elgrand', 'G 55 AMG', 'Passat 2.0 tfsi', 'QX56', 'Vectra B', 'Defender 90 Cabrio', 'Niro', 'Highlander 2,4', '320 2.2', '428', 'Dart GT 2.4', 'Zafira B', '525', 'Prius C YARIS IA', 'ML 350 4 MATIC', 'Ist', 'Panamera', '500 SPORT', 'Golf 6', '1300', 'C 300 sport', 'Equinox LT', 'Land Cruiser', 'Jetta SEL', '159', 'A4 Sline', '340', 'Polo', 'Caliber sxt', 'GL 450', 'GTI', 'Impreza', 'Corolla S', 'G 320', 'Cooper S Cabrio', 'Clio', '300', 'C 230', 'ML 500 AMG', 'XJ', 'A 160', 'Avalon', 'Legend FULL', '335', 'F-pace', '300 LIMITED', 'Veloster', '320 320', 'CX-3', '550', 'Octavia', '500X Lounge', 'Impreza WRX/STI LIMITED', 'Touareg', 'Focus SE', 'Kicks', 'RX 450 H', 'Venza', 'Demio', 'Atenza', 'Elysion 3.0', 'Pathfinder', 'Juke juke', 'Fred', 'Tucson SE', '630', 'X3', 'Cherokee', 'Camry XLE', 'Mariner', '500 Abarth', 'Outlander sport', 'Crossfire', 'C 250 A.M.G', 'ISIS', 'Edge', 'Rx-8', 'Musa', 'Transit 2.4', 'Traverse', 'Fusion HYBRID', 'E 300 AVANTGARDE-LTD', 'C4', 'Eunos 500', 'C 240', 'X5 M', 'RS7', 'Omega', 'IS 350', 'Prius plagin', 'Bluebird', 'Cruze strocna', 'FX35', 'Demio evropuli', 'S 500 67', 'Rogue', 'Cooper CLUBMAN', 'Cayman', 'X1', 'FIT Hbrid', 'Will Vs', 'A5', 'HHR', 'Escape Hybrid', 'TLX', 'S40', 'i40', 'Grandis', '500 turbo', 'Panamera S', 'Astra 1600', '500 Sport', 'Journey', 'G 350', 'Mazda 3', 'CLS 350', 'EcoSport', 'Mirage', 'Wish', 'H2', 'Prius plug-in', 'Crosstrek', 'Sprinter 315CDI-XL', 'Astra G', 'Cruze Cruze', 'Grand Cherokee', 'Vito 2.2', 'Accord', 'FIT LX', 'ML 55 AMG', 'A 140 140', 'Auris', '280', 'X5 XDRIVE ', 'Countryman', 'Patriot', 'Yukon', 'Sonic LT', 'xD', 'ML 350 ML350', 'RX 300', 'Panda', 'GL 550', 'Passport', 'A6 UNIVERSAL', 'CLK 320 avangarde', '2106', 'C 230 2.0 kompresor', 'Picanto', 'Edix', 'Azera', 'Passat sel', '525 i', 'Pathfinder SE', 'Ipsum', 'CT F sport', 'X5 DIESEL', 'Transit Custom', '328 i', 'Patrol', 'Step Wagon RG2 SPADA', 'Quattroporte', '335 D', 'Tourneo Connect', 'MPV LX', '2121 (Niva)', 'Fusion TITANIUM', 'C 200 7G-TRONIC', 'Camry S', 'MPV', 'Carnival grand', 'S 500', 'Cruze RS', 'Cadenza', 'Elantra GT', 'FIT HIBRID', 'Jetta SE', 'Q3', 'E 230', 'Integra', 'Fit Aria', 'Passat SEL', 'Quest 2016', 'A4 B5', 'Q7 sport', 'E 280 CDI', 'Fusion Titanium', 'Passat sport', 'Camry XSE', 'Crafter 2.5 TDI', '250', 'Vitz funkargo', 'PT Cruiser', 'ML 500', 'Durango', 'CLK 240', 'Megane', 'Town Car', 'Sprinter 313', 'Caldina', 'A4 S line', 'GS 350', 'Sonata 2.4L', 'Stream', 'Liana', '206', 'Samurai', 'Astra td', 'Vectra H', 'Sprinter 311', 'Sienta LE', '230W153', 'Prius V HYBRID', 'Meriva', '3.20E+38', 'S60', 'Expedition', '500 sport', 'Belta', 'CLK 55 AMG', 'Taurus interceptor', 'Getz', '230', 'Ipsum S', 'Presage RIDER', 'Prius C ', '31105', 'Carisma', 'Golf 2', 'S 430', 'Forester SH', 'Vectra b', 'Volt premier', 'Camry sport', 'A 170 Avangard', 'S3', 'Fuga', 'Aqua S', 'Continental', 'GLK 350', 'E 280', 'Cruze sonic', 'RX 350 F sport', 'Sonata SE', 'Explorer', 'T5', 'Twingo', '650 450 HP', 'X1 X-Drive', 'Demio Sport', 'ML 320 AMG', 'M4 Competition', 'Golf TDI', 'Pajero Mini', 'CLK 230 .', 'Q5 S-line', 'CLK 270', 'A4 premium', 'Grandeur', 'Elantra limited', 'E 220 cdi', 'X3', 'B9 Tribeca', 'Micra', '114', 'Vito 113', 'GL 350', 'E 36 AMG', 'Astra j', 'Optima X', 'Caddy', 'Omega 1', 'Forester stb', 'RS6', 'Bora', '270', '550 GT', 'CLS 550', 'Grand Cherokee Saiubileo', 'Navigator', 'E 280 3.0', 
'Ceed', 'Every Landy NISSAN SEREN', 'Mustang ecoboost', 'Land Rover Sport', 'A4 B7', '645', 'Fabia', 'Forte', 'Elantra SE', 'Escape HYBRID', 'Prius 2014', '535 535', 'S 400', 'Fusion Bybrid', 'G 550', 'Corolla spacio', 'ML 280', 'VOXY 2003', 'Silvia', '323', 'Tiguan', 'C-MAX SE', 'Courier', 'Charger RT', 'Minica', 'Teana', 'tC', 'Discovery', 'Golf 3', 'JX35', '428 Sport Line', 'Verisa 2007', 'Accent GS', 'Cruze PREMIER', '9-Mar', 'Fusion hybrid', 'M37', 'XC90 2.5turbo', 'Versa SE', '130', 'Fortuner', 'Town and Country', '535 M', 'Sprinter 411', 'Chariot', 'Carnival', 'A 170', 'CLS 63 AMG', 'Golf Gti', 'Step Wagon Pada', 'Focus TITANIUM', 'FIT Modulo', '118 2,0', 'i3', 'Vitz RS', 'CLK 430', 'Tucson Limited', 'KA', 'Fusion SE', 'Omega c', 'Volt PREMIER', 'Mazda 6 TOURING', 'Fred HIBRIDI', 'CL 500', '220', '535 i', 'Astra astra', 'Focus se', 'Prius prius', 'Paceman', 'Optima SXL', '420', 'R2', 'Mazda 3 SPORT', 'Sienta', 'X5 ', 'Veracruz', 'Elantra sport limited', 'Sorento SX', 'Santa FE long', 'C 200 2.0', 'X4', 'Elantra LIMITED', 'Eclipse', 'Smart Fortwo', 'CX-5 Touring', 'Tiida AXIS', 'Pajero MONTERO', 'E 500 AVG', 'Aerio SX', '118', 'Colorado', 'S 350', 'SRX', 'Wingroad', 'ColtPlus', '550 F10', 'Sambar', 'Kangoo Waggon', 'GLE 63 AMG', 'Outlander 2.0', 'Vesta', 'V 230', 'Cruze ltz', 'C 63 AMG', 'Octavia Scout', 'Prius S', 'Corvette', 'Prius C 80 original', 'X5 rest', 'Optima k5', 'M5', 'Vectra 1.6', 'Niva', 'Discovery LR3', 'RAV 4 L', '1111', 'Elantra 2014', 'Transit Fff', 'FIT PREMIUMI', 'Legacy B4 twin turbo', 'Vitz i.ll', 'Freelander', '645 CI', 'Aqua HIBRID', 'M4', 
'Optima HYBRID', 'TSX', 'CR-Z', 'Countryman S', 'Premacy', 'Avalanche', 'Santa FE sport', 'Elantra i30', 'E 350 AMG', 'H3', 'Skyline 4WD', '530 525i', 'IS-F', 'Allroad', 'Yaris IA', 'Mondeo', 'M550', 'Agila', 'E 220 CDI', 'Tiguan SE', 'Neon', 'Jetta TDI', 'Panamera GTS', 'FIT Hybrid', 'RDX', 'X1 4X4', 'C 320 CDI', 'Accord CL9 type S', 'Focus ST', 'Prius Plug IN', 'Octavia SCOUT', 'Lancer', 'Camry SPORT PAKET', 'Vito 110d', 'GLE 450', 'Cooper F-56', 'Outback 2007', 'Sonata Hybrid', 'Combo 1700', 'Elantra Gt', 'XC90', 'Seicento fiat 600', 'X3 SDRIVE', 'Jetta 1.4 TURBO', 'Camry SPORT', 'CTS', 'Sonic', 'R 320', '416', 'C30', '316 i', 'Harrier', 'GLK 250', 'Berlingo', '525 ///M', 'Nubira', 'QX80', 'Elantra GS', 'Optima hybid', '530 I', 'X5 M packet', 'Kalos', 'Sintra', 'Hr-v', 'G37', '100', 'Envoy', 'E 430', 'B 170', 'X 250', 'Prius ', 'Golf 1.8', 'C 250 1,8 turbo', 'Golf GOLF 5', 'GLE 400 Coupe, AMG Kit', 'Stella', '31514 UAZ', 'SL 55 AMG', 'Crafter', 'FIT ex', 'E 500 AMG', '118 M-sport LCI', 'C 230 kompresor', 'Versa s', 'C5', 'Sai', 'Duster', 'CT  F-sport', 'Elantra 2016', 'Wrangler sport', 'Sorento EX', 'Vito Extralong', 'Land Cruiser 100', 
'500 Lounge', 'RAV 4 Dizel', 'ML 350 3.7', 'E 270 AVANGARDI', 'Corolla 140', '525 525', 'Passat SE', 'E 200 2000', 'March 231212', '730', 'Allroad Allroad', 'Patriot 70th anniversary', 'G35 x', 'Countryman s', 'Trax', '535 comfort-sport', 'FIT fit', 'Veloster TURBO', 'Sprinter 516', 'Verisa', 'GLC 300', 'C 180 2.0', 'Caravan', 'Montero', 'C 32 AMG', 'Jetta sei', '616', 'Carens', '435', '3.25E+48', 'HS 250h Hybrid', 'Viano', 'Omega b', 'REXTON SUPER', 'S6', 'IX35', '2140', 'Cooper r50', 'Leaf', 'Sprinter Maxi-Ã©Â\x86Â¿?Max', 'E 400', 'Cerato K3', 'LAFESTA', 'Highlander 2.4 lit', 'Forester cross sport', '328 sulev', 'Q45', 'Vectra C', 'Crosstour', 'RAM 1500', 'Passat tdi sel', 'Passat pasat', 'CC R line', 'XK', 'Kangoo', 'ML 350 SPECIAL EDITION', 'Camry sporti', 'Highlander sport', 'Avalon limited', 'C 250 AMG', 'Transit T330', 'R 350 BLUETEC', 'Mustang cabrio', 'E 350 4 Matic AMG Packag', 'Sonata hybrid', 'E 250', 'Sonata Limited', 'Eos', 'Land Cruiser PRADO', 'Allante', 'X1 28Xdrive', 'Outlander Sport', 'Outlander SPORT', 'Silverado', 'Caravan tradesman', 'Outback Limited', 'Prius C Hybrid', 'Camry XLEi', 'C 270', 'A5 Sportback', 'Lancer GTS', 'Hr-v EX', 'LATIO', 'FIT SPORT', 'Vaneo', 'Millenia', '225', 'Enclave', 'V50', 'Camry sport se', 'S 350 W2222', '208', 'GLC 300 GLC coupe', '530 G30', '401', 'B-MAX', 'Sportage PRESTIGE', 'GL 350 Bluetec', 'Jetta sport', 'Vitara GL+', 'X6 M', 'Veloster Turbo', 'Transit CL', 'Sierra DIZEL', 'Jetta 2', 'Lupo', 'ML 270', 'I', 'Sprinter 316', 'GS 300', 'Range Rover Vogue', 'Pajero Sport', 'M6', 'Prius s', 'A6 premium plus', 'Camry Se', 'TL', '207', 'Passat B5', 'Fusion HIBRID', 'Prius C 2013', 'A 200', 'Forester L.L.BEAN', 'Hilux Surf', '650', 'Hr-v EXL', 'Insight EX', 'Focus Titanium', '550 M Packet', 'Forester ', '406', 'ML 250', 'Daimler', 'Frontier', 'G20', 'X5 4', '730 3.0', 'Vitara', '320 2.0', 'Sprinter VIP CLASS', 'RX 450 HYBRID', 'A6 QUATTRO', '535 Twinturbo', 'A 170 CDI', 'Rogue SPORT', 'Focus Fokusi', 'Camaro LS', 'Megane GT Line', 'Land Cruiser 80', '290', 'LX 570', 'Tribute', '528 3.0', 'X-type', 'Sonata HYBRID', 'BRZ', '745', 'Navara', 'S-type', 'Polo GTI ', '5.30E+62', 'Tigra', 'Wrangler ARB', 'Matrix XR', '135', 'E 200 CGI', 'Fiesta SE', '500 s', 'CC 2.0 T', 'F50', 'RIO lX', 'Jetta Hybrid', 'A3 PREMIUM', 'Optima ECO', 'Corolla ECO', 'Elantra GLS / LIMITED', '969 luaz', 'E 350 4 MATIC', 'Prius 3', 'Z4', 'Cruze LS', 'Escalade', 'Cougar', 'Alto Lapin', '320 I', 'Malibu eco', 'CLK 200 200', 'Micra <DIESEL>', 'TL saber', 'Escape escape', 'GLE 400', 'GLS 450', 'E 230 124', 'Fusion 2015', 'Escape Titanium', 'B 170 B Class', 'Trailblazer', 'IX35 2.0', 'C 250 1.8', 'RC F', 'Ranger Wildtrak', 'GLA 200', 'F-type', 'Outlander xl', 'Leon', 'kona', 'RAV 4 XLE', 'Hiace', 'Impreza Sport', 'Camry SE HIBRYD', 'X6 GERMANY', 'Acadia', '206 CC', 'CLK 200 Kompressor', 'Prius 11', 'S-max', 'Focus Flexfuel', 'Civic EX', '400', 'Elantra Se', 
'GS 450', '320 DIESEL', 'Frontera A B', 'EX37', 'Megane 1.9CDI', 'Kizashi sporti', 'C-MAX SEL', 'Continental GT', 'Galaxy', 'NEW Beetle', 'GLS 63 AMG', 'CLS 350 AMG', 'Cruze LTZ', '3110', '500C', '540 I', 'Range Rover VOGUE', 'FIT RS MUGEN', 'XL7', 'Celica', 'A3 4X4', 'Cayenne s', 'GLE 43 AMG', 'Tiida 2008', 'X5 restilling', '960', 'X5 3.0', 'CLS 450 CLS 400', 'Noah','3008', 'Captur QM3 Samsung', 'EX35', 'Suburban', '32214', 'Lupo iaponuri', 'IS 350 C', 'Civic Ferio', 'Optima Hybrid', 'March Rafeet', 'C 400', 'Armada', 'H6', 'Aqua G', 'Countryman S turbo', 'Demio 12', 'X-Trail NISSAN X TRAIL R', 'Optima EX', 'Mark X Zio', 'Ractis', 'C 300 4matic', 'C30 2010', 'S 550 LONG', 'Countryman sport', 'Highlander limited', 'Volt Premier', 'Crossroad',  'Jetta SPORT', '500 sport panorama', 'Focus SEL', 'Sonata 2.0t', 'Sportage EX', 'Ranger', 'E 240 E 240', '3.18E+38', 'S 350 Longia', 'Sharan', 'SLK 230', 'Sportage SX', 'Sonata Sport', 'X5 x5', '735', 'Vito long115', 'ML 350 sport', 'MDX', 'Prius C hybrid', 'X5 XDRIVE', 'CLK 350', 'C 43 AMG', 'Camry HYBRID', 'RAV 4 XLE Sport', 'Range Rover Evoque', 'Touran', 'CLS 550 550', 'Transit ', 'Transporter', 'Legacy b4', 'Caliber journey', '607', 'GX 470 470', 'Escort', '100 NX', 'Passat tsi-se', 'Yaris SE', 'Patriot Latitude', 'Elantra gt', 'Sonata LPG', 'Vito Exstralong', 'Legacy bl5', 'Vito 111', 'Primera', 'E 200 w210', 'CLA 45 AMG', 'Prius 9', 'Astra GE', 'Juke Juke', 'C 240 w203', 'Legacy B4', 'Sonata SE LIMITED', 'Passat B7', 'Mark X', 'Astra suzuki mr wagon', '166', '311', 'Grand HIACE', 'Passat RLAINI', '626', 'FIT "S"- PAKETI.', 'Prius C Navigation', 'Prius plugin', 'Galloper', '640 M', 
'E 270 CDI', 'Estima', 'Escape SE', 'X5 e53', 'E-pace p200', 'Feroza', 'C70', '525 TDI', 'Patrol Y60', 'QX60', 'Galant', 'CLS 350 JAPAN', 'CERVO', 'GL 350 BLUTEC', 'S70', 'Vito Extra Long', 'Colt Lancer', '318 m', 'Crafter 2,5TDI', 'CLK 280', 'S 430 4.3', 'Insight LX', 'Sierra', 'ColtPlus Plus', 'XV LIMITED', 'X5 ', 'Cruze Premier', 'Juke Nismo RS', 'RX 400 H', 'Megane 5', 'Presage', 'Almera dci', 'E 260', 'FIT PREMIUM PAKETI', 'Aqua L paketi', 'Sprinter EURO4', 'RAV 4 SPORT', 'Rasheen', 'A4 premium plius', 'Vue', 'Panamera 4', 'CC sport', '535 XI', 'GLE 400 A M G', 'RX 400 RESTAILING', '116', 'IS 300', 
'216', 'E 350 w211', 'Mazda 6 Grand touring', 'Almera', 'Mazda 6 Grand Touring', '1500,1600 Schtufenheck', 'C 200 Kompressor', 'AMG GT S', 'Sprinter 314', 'RX 400 HYBRID', 'Astra A.H', 'Sonata LIMITED', 'RAV 4 s p o r t', 'Will Chypa', '2111', 'FIT Premiym', 'Superb', 'Prius C aqua', 'Lantra', 'Civic Hybrid', 'Terrano', 'H1 starixs', 'Prius TSS LIMITED', 'Veloster remix', 'Giulietta', 'Passo', 'Cruze S', 'Tiida Latio', 'Lancer GT', 'Rogue SL', 'G 65 AMG G63 AMG', 'Sonata S', 'X6 40D', 'FIT GP-5', 'F-type R', 'Jetta se', 'Jetta GLI', 'X-Trail X-trail', 'Axela', 'Jetta sel', 'C-MAX PREMIUM', 'Transit 135', 'Vito 111 CDI', 'Protege', 'Tucson Se', 'SJ 413 Samurai', 'Inspire', '320 M', 'Mira', 'Fusion HYBRID SE', 'IS 250 TURBO', '520 I', 'ML 270 CDI', 'X5 4.8is', 'Aqua g soft leather sele', 'E 350 4 matic', 'S 500 long', 'Grand Cherokee special e', '120', '7.30E+34', 'Range Rover Evoque 2.0', '530 M', 'Mariner Hybrid', '969 968m', 'Sebring', 'Wizard', 'RAV 4 Le', 'E 350 212', '740 i', 'C-MAX C-MAX', 'A7 Prestige', 'C 230 2.5', 'Skyline GT250', 'Sprinter 316 CDI', 'Explorer XLT', 'Prius BLUG-IN', 'CLK 200 208', 'Phantom', 'ML 350 BLUETEC', 'Outback 3.0', 'X5 Sport', 'Cooper S', 'Aqua sport', 'Vito 115', '940', 'RIO lx', 'Prius Plug in', 'A6 C7', 'Sonata blue edition', 'X-Trail gt']

manufacturer_options = ['MERCEDES-BENZ', 'SSANGYONG', 'HYUNDAI', 'LEXUS', 'BMW', 'OPEL', 'KIA', 'VOLVO', 'NISSAN', 'TOYOTA', 'HONDA', 'FORD', 'FIAT', 'VOLKSWAGEN', 'MAZDA', 'CHEVROLET', 'AUDI', 'SUZUKI', 'MITSUBISHI', 'SUBARU', 'LINCOLN', 'UAZ', 'DODGE', 'BUICK', 'JEEP', 'VAZ', 'DAEWOO', 'PORSCHE', 'DAIHATSU', 'GAZ', 'JAGUAR', 'CITROEN', 'ISUZU', 'MINI', 'ROVER', 'GMC', 'CHRYSLER', 'LAND ROVER', 'MASERATI', 'CADILLAC', 'PEUGEOT', 'RENAULT', 'SCION', 'INFINITI', 'ALFA ROMEO', 'SKODA', 'MERCURY', 'LANCIA', 'ACURA', 'HUMMER', 'SAAB', 'GREATWALL', 'MOSKVICH', 'FERRARI', 'ZAZ', 'SEAT', 'BENTLEY', 'HAVAL', 'SATURN', 'ROLLS-ROYCE']



fuel_type_options = ['Petrol', 'Diesel', 'LPG', 'Hybrid', 'CNG', 'Plug-in Hybrid', 'Hydrogen']
leather_options = ['Yes', 'No']
category_options =  ['Sedan', 'Jeep', 'Minivan', 'Coupe', 'Universal', 'Hatchback', 'Goods wagon', 'Microbus', 'Pickup', 'Cabriolet', 'Limousine']
cylinders_options = [4,6,8,5,10,2,1,9,12,3,7,16]
color_options=['Beige', 'Black', 'Blue', 'Brown', 'Carnelian red', 'Golden', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'Silver', 'Sky blue', 'White', 'Yellow']
doors_options=['4' ,'2' ,'>5']


input_data = {    
    'Airbags': st.selectbox('Airbags', air_bag_options),
    'Prod. year': st.number_input('Production Year', min_value=1900, max_value=2025, value=2015),
    'Mileage': st.number_input('Mileage', min_value=0, value=100000),
    'Drive wheels': st.selectbox('Drive wheels', drive_wheels_options),
    'Wheel': st.selectbox('Wheel position', wheel_options),
    'Gear box type': st.selectbox('Gear box type', gear_box_options),
    'Manufacturer': st.selectbox('Manufacturer', manufacturer_options),
    'Model': st.selectbox('Model', model_options),
    'Levy': st.number_input('Levy', min_value=0, value=0),
    'Fuel type': st.selectbox('Fuel type', fuel_type_options),
    'Leather interior': st.selectbox('Leather interior', leather_options),
    'Category': st.selectbox('Category', category_options),
    'Cylinders': st.number_input('Cylinders', min_value=1, max_value=16, value=4), 
}

feature_columns = ['Airbags', 'Prod. year', 'Mileage', 'Drive wheels', 'Wheel',
                     'Gear box type', 'Manufacturer', 'Model', 'Levy', 'Fuel type',
                     'Leather interior', 'Category', 'Cylinders']
input_data['Airbags']=input_data['Airbags']
input_data['Prod. year']=input_data['Prod. year']
input_data['Mileage']=input_data['Mileage']
input_data['Drive wheels'] = drive_encoder.transform(pd.DataFrame([[input_data['Drive wheels']]], columns=['Drive wheels']))[0][0]
input_data['Wheel'] = wheel_encoder.transform(pd.DataFrame([[input_data['Wheel']]], columns=['Wheel']))[0][0]
input_data['Gear box type'] = gear_encoder.transform(pd.DataFrame([[input_data['Gear box type']]], columns=['Gear box type']))[0][0]
input_data['Manufacturer'] = manufacturer_encoder.transform(pd.DataFrame([[input_data['Manufacturer']]], columns=['Manufacturer']))[0][0]
input_data['Model'] = model_encoder.transform(pd.DataFrame([[input_data['Model']]], columns=['Model']))[0][0]
input_data['Levy']=input_data['Levy']
input_data['Fuel type'] = fuel_encoder.transform(pd.DataFrame([[input_data['Fuel type']]], columns=['Fuel type']))[0][0]
input_data['Leather interior'] = leather_encoder.transform(pd.DataFrame([[input_data['Leather interior']]], columns=['Leather interior']))[0][0]
input_data['Category'] = category_encoder.transform(pd.DataFrame([[input_data['Category']]], columns=['Category']))[0][0]
input_data['Cylinders']=input_data['Cylinders']



input_df = pd.DataFrame([input_data], columns=selected_features)

input_df_scaled = pred_scaler.transform(input_df)
input_df_scaled = pd.DataFrame(input_df_scaled, columns=selected_features)



prediction = pred_model.predict(input_df_scaled)
# st.subheader(f"Prediction using {model_option}:")
# st.write(f"Predicted Price: ${prediction[0]:,.2f}")
if st.button(f"Predict"):
    st.subheader(f"Prediction using {model_option}:")
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")


st.subheader(f"{model_option} Test Accuracy:")
st.progress(test_r2) 
st.write(f"Test Accuracy: {test_r2:.4f}")

