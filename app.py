import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import base64
import numpy as np
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from pymongo import MongoClient
from urllib.parse import quote_plus
from PIL import Image
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from io import BytesIO
from keras.src.applications.inception_v3 import InceptionV3
import tensorflow as tf


""" Creating the instance of the flask application """
app = Flask(__name__)


# Secret key for session management
app.secret_key = 'elon-musk'

UPLOAD_FOLDER = 'media'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Replace the following with your MongoDB Atlas connection details
username = "OnkIndustries"
password = "Asus15@9527"
cluster_name = "onkcloud.8y9wfls.mongodb.net"
database_name = "fish_Freshness_Prediction"  # Replace with your actual database name
collection_name = "fish_data"  # Replace with your actual collection name
# Escape username and password
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)

# Create a MongoClient using the connection string
connection_string = f"mongodb+srv://{escaped_username}:{escaped_password}@{cluster_name}/{database_name}?retryWrites=true&w=majority"
client = MongoClient(connection_string)
# Access the database
db = client[database_name]
# Access the collection
collection = db[collection_name]  # Replace 'fish_data' with your actual collection name


""" Machine learning Models """
""" Works Fine : Predicts if the image is fish or not , Returns Integer """

def ifFishOrNot(image_data):
    # Check if image_data is None
    if image_data is None:
        print("Error: image_data is None.")
        return -1  # Or any other appropriate error code or value

    species_list = ['Gourami', 'Midnight Lightning Clownfish', 'greenling', 'sailfish', 'Rockfish',
                    'bearded fireworm', 'sabertooth', 'leafy seadragon', 'grunt', 'Royal Gramma', 'Wolffish',
                    'Red Emperor Snapper', 'Golden Tilefish', 'pompano', 'Tetra', 'Croaker', 'whiting',
                    'dwarf gourami', 'salmon', 'Viperfish', 'sailfin catfish', 'white catfish', 'rougheye rockfish',
                    'yellowtail', 'Banded Wrasse', 'Golden Dorado', 'wrasse', 'Green Clown Goby', 'Plecostomus',
                    'flounder', 'Black Marlin', 'flathead', 'Sea Robin', 'Chalk Bass', 'redtail catfish',
                    'Candy Basslet', 'Bullhead', 'hammerhead shark', 'cutlassfish', 'Diamond Watchman Goby',
                    'brill', 'bluntnose knifefish', 'zooanthid', 'glass knife fish', 'Indian Threadfish',
                    'rockfish', 'tilefish', 'mantis shrimp', 'jackfish', 'Orangespine Unicornfish', 'Rasbora',
                    'halfbeak', 'snook', 'oyster', 'Lyretail Anthias', 'Sixbar Wrasse', 'Betta Fish', 'Bigeye Scad',
                    'Medusafish', 'Yellowtail', 'Arowana', 'shrimp', 'Mandarinfish', 'zebrapleco', 'Kingfish',
                    'Firefly Squid', 'Parrotfish', 'Celestial Pearl Danio', 'pimelodid catfish', 'lobster',
                    'Rainbow Runner', 'Haddock', 'squid', 'Snapper', 'sea squirt', 'Bicolor Blenny', 'Wahoo',
                    'Longnose Butterflyfish', 'dolly varden trout', 'Purple Tang', 'tilapia', 'Giant Oarfish',
                    'whitefish', 'Piranha', 'sea slug', 'Turbot', 'Snowflake Clownfish', 'skate', 'bluefin tuna',
                    'blowfish', 'Ocellaris Clownfish', "Kaudern's Cardinalfish", 'butterflyfish', 'pike',
                    'snakehead', 'mosquitofish', 'Handfish', 'sea butterfly', 'King Threadfin', 'hagfish',
                    'Butterfish', 'unicornfish', 'Longfin Tuna', 'Archer Catfish', 'Mangrove Snapper', 'arapaima',
                    'sculpin', 'Halibut', 'Cichlid', 'Jewel Damselfish', 'Tiger Shovelnose Catfish',
                    'Frostbite Clownfish', 'Anchovy', 'octopus', 'Zebra Pleco', 'Diamond Goby', 'whiskerfish',
                    'tadpole', 'barramundi', 'Eel', 'searobin', 'Percula Clownfish', 'herring', 'Regal Angelfish',
                    'zebraple', 'Yellow Tang', 'sea spider', 'yellow perch', 'Koran Angelfish', 'manta ray',
                    'sturgeon', 'coral trout', 'Rock Beauty', 'Yellowbelly Flounder', 'Salmon', 'Atlantic Herring',
                    'mahi-mahi', 'triggerfish', 'Leafy Sea Dragon', 'long-whiskered catfish', 'Turquoise Killifish',
                    'Pacific Hake', 'Lined Seahorse', 'cuttlefish', 'Anthias', "Scott's Velvet Fairy Wrasse",
                    'Angelfish', 'sweeper', 'platy', 'stonefish', 'Queen Triggerfish', 'Emperor Angelfish',
                    'whale catfish', 'Blind Goby', 'Tube-eye', 'frilled shark', 'dragonet',
                    'Yellowspotted Trevally', 'zamurito', 'Copperbanded Butterflyfish', 'ling', 'Flame Angelfish',
                    'Glass Knifefish', 'bullhead', 'lamprey', 'Yellowstripe Scad', 'grayling', 'Redfish',
                    'anglerfish', 'zonetail', 'brycon', 'Atlantic Mackerel', "Randall's Goby", 'Zebrafish',
                    'Peacock Flounder', 'Monkfish', 'wolf fish', 'Koi', 'Mahi-mahi', 'cichlid', 'halibut',
                    'warty sea cucumber', 'surgeonfish', 'bichir', "lion's mane jellyfish", 'leaffish',
                    'vampire fish', 'Perch', 'snapper', 'sea lily', 'Barrel-eye', 'Cobia', 'Giant Snakehead',
                    'Scooter Dragonet', 'urchin', 'zebrashark', 'Indian Ocean Sailfin Tang',
                    'Golden Head Sleeper Goby', 'water flea', 'axolotl', 'mola mola', 'Trout', 'Lanternfish',
                    'Corydoras Catfish', 'Leopard Whipray', 'Longfin Escolar', 'Midas Blenny', 'stingray',
                    'Bigfin Reef Squid', 'Redtail Catfish', 'sea cucumber', 'zander', 'starfish', 'frog', 'arawana',
                    'sea anemone', 'Lemonpeel Angelfish', 'Rainbowfish', 'Sardine', 'trevally', 'weeverfish',
                    'mudfish', 'platydoras', 'Blue Tang', 'Stoplight Loosejaw', 'Bicolor Pseudochromis', 'plaice',
                    'Chinook Salmon', 'Dolphin Fish', 'shark', 'Red Sea Sailfin Tang', 'pollock', 'crab',
                    'john dory', 'Wrasse', 'clownfish', 'angel shark', 'Fire Goby', 'ziggies', 'pollack', 'eel',
                    'Bass', 'Barracuda', 'redfish', 'zugzug', 'Albacore Tuna', 'Archerfish', 'Pacific Cod',
                    'Elephantnose Fish', 'Foxface Rabbitfish', 'betta', 'Tilefish', 'anchovy', 'Lawnmower Blenny',
                    'newt', 'pickerel', 'bream', 'Pinecone Fish', 'Spanish Mackerel', 'glass catfish',
                    'Giant Trevally', 'Majestic Angelfish', 'barracuda', 'Pufferfish', 'Fourspot Butterflyfish',
                    'electric catfish', 'Sixline Wrasse', 'Pollock', 'mackerel', 'bristle mouth', 'snail',
                    'talking catfish', 'wolffish', 'barnacle', 'Colossal Squid', 'jawfish', 'tubeblenny',
                    'Yellow Banded Pipefish', 'Neon Dottyback', 'blue tang', 'Bluebanded Goby', 'Threadfin Bream',
                    'Barramundi Fish', 'Grouper', 'Yellowfin Surgeonfish', 'pencil catfish', 'smelt', 'scallop',
                    'chub', 'parrotfish', 'rainbow trout', 'Boarfish', 'Black Ice Ocellaris Clownfish',
                    'Gulper Eel', 'tigerfish', 'Jackfish', 'Bumblebee Goby', 'caecilian', 'knifefish',
                    'Flame Hawkfish', 'Molly', 'flagfish', 'Lingcod', 'blind cavefish', 'gudgeon',
                    'Yasha White Ray Shrimp Goby', 'Herring', 'stickleback', 'lions mane jellyfish', 'pufferfish',
                    'Marlin', 'sea urchin', 'sheatfish', 'zanderfish', 'koi', 'torsk', 'Rainbow Trout',
                    'Fiji Blue Devil Damsel', 'Twinspot Goby', 'Spotfin Hogfish', 'Red Mandarin Dragonet',
                    'pipefish', 'Glassfish', 'Banggai Cardinalfish', 'krill', 'Freckled Hawkfish',
                    'Powder Blue Tang', 'Gulf Menhaden', 'Spotted Grunt', 'zorsefish', 'goliath tigerfish',
                    'swordtail', 'tang', 'Tigerfish', 'Green Chromis', 'lungfish', 'Goldfish', 'Longnose Gar',
                    'bluefish', 'Yellow Eye Kole Tang', 'moonfish', 'Slippery Dick', 'Humboldt Squid', 'seahorse',
                    'Japanese Amberjack', 'Orchid Dottyback', 'Flagfish', 'Pompano', 'bobbit worm', 'Killifish',
                    'boxfish', 'paddlefish', 'Bonito', 'Ornate Leopard Wrasse', 'Threadfin Geophagus',
                    'electric ray', 'Pajama Cardinalfish', 'prawn', 'Red-bellied Piranha', 'guppy', 'lionfish',
                    'zebrafish', 'Stargazer', 'Dusky Grouper', 'Discus', 'Bluestreak Cleaner Wrasse',
                    'piraiba catfish', 'Green Jobfish', 'Dragon Wrasse', 'African Pike', 'Clown Loach',
                    'Yellow Pyramid Butterflyfish', 'moon jellyfish', 'dogfish', 'tinfoil barb', 'Red Fire Goby',
                    'Jellybean Parrotfish', 'Dragonet', 'Yellow Banded Possum Wrasse', 'thresher shark', 'Cod',
                    'Blue Spot Jawfish', 'Triggerfish', 'whitefin wolf fish', 'zorse', 'Clown Knifefish',
                    'Yellowtail Damselfish', 'gurnard', 'threadfin', 'Pike', 'Orange Spotted Goby', 'Bristlemouth',
                    'Orange Roughy', 'Glass Squid', 'Silver Carp', 'marlin', 'Ribbonfish', 'Psychedelic Mandarin',
                    'Shortraker Rockfish', 'Cherubfish', 'arowana', 'Clown Coris Wrasse', 'sea angel', 'Sturgeon',
                    'Silver Arowana', 'tarpon', 'Amberjack', 'Blanket Octopus', 'Betta', 'tigerperch', 'sea fan',
                    'muskellunge', 'striped bass', 'fluke', 'Electric Blue Hap', 'Ribboned Seadragon',
                    'Blueface Angelfish', 'goblin shark', 'zebra oto', 'brittle star', 'leopardfish', 'Neon Tetra',
                    'Golden Snapper', 'glassfish', 'Gulper Shark', 'isopod', 'comb jelly', 'clam', 'Chimaera',
                    'trout', 'Carp', 'monkfish', 'Achilles Tang', 'giant danio', 'Japanese Eel', 'Bluefish',
                    'weedy seadragon', 'Powder Brown Tang', 'Yellow Watchman Goby', 'Darter', 'Batfish', 'mullet',
                    'chambered nautilus', 'piranha', 'Mandarin Goby', 'bluegill', 'Bluestripe Snapper',
                    'soft coral', 'Pink Salmon', 'longfin smelt', 'Shortnose Greeneye', 'Bluefin Trevally',
                    'Bobtail Squid', 'electric eel', 'Moorish Idol', 'coelacanth', 'sawfish', 'slug',
                    'silver dollar fish', 'drumfish', 'catfish', 'Goby', 'scorpionfish', 'killifish', 'swordfish',
                    'Razorfish', 'Yelloweye Rockfish', 'Hooded Fairy Wrasse', 'wahoo', 'Yellowfin Croaker', 'cod',
                    'Red Drum', 'Mocha Storm Clownfish', 'argentine blue-bill', 'amphipod', 'pangasius',
                    'hard coral', 'Flounder', 'Drum', 'sun catfish', 'goby', 'gourami', 'tailspot blenny', 'hake',
                    'squatina', 'Vampire Squid', 'mussel', 'Koi Fish', 'Blackfin Tuna', 'Weedy Sea Dragon',
                    'Tilapia', 'angelfish', 'nudibranch', 'zebraperch', 'walleye', 'Hogfish', 'Horse Mackerel',
                    'Copperband Butterflyfish', 'Vlamingii Tang', 'Tuna', 'Fairy Wrasse', 'carp', 'basket star',
                    'sunfish', 'giraffe catfish', 'Harlequin Tuskfish', 'firefly squid', 'Swordtail', 'ribbon eel',
                    'Bigeye Tuna', 'Barb', 'sheepshead', 'medusafish', 'haddock', 'Swordfish', 'Snakehead',
                    'rainbowfish', 'trumpetfish', 'spookfish', "McCulloch's Clownfish", 'Pineapplefish', 'Platy',
                    'Atlantic Cod', 'Guppy', 'Skate', 'Blue Throat Triggerfish', 'goldfish', 'sardine',
                    'Yellow Coris Wrasse', 'Ghost Pipefish', 'batfish', 'Gurnard', 'zooids', 'Boxfish', 'garfish',
                    'Queen Parrotfish', 'zoogoneticus', 'Blue Hippo Tang', 'perch', 'Melanurus Wrasse',
                    'zebra pleco', 'giant trevally', 'wallago catfish', 'grouper', 'Mackerel', 'snipefish',
                    'Domino Damsel', 'sargo', 'Japanese Swallowtail Angelfish', 'Catfish', 'Tripodfish',
                    'feather star', 'placidochromis', 'Powder Blue Surgeonfish', 'Corydoras', 'tuna', 'zoarcid',
                    'Alaskan Pollock', 'Tube Snout', 'plecostomus', 'hoki', 'spotted talking catfish', 'ribbonfish',
                    'coral', 'Tiger Jawfish', 'sole', 'Paper Nautilus', 'salamander', 'gar', 'Yellow Perch',
                    'turbot', 'Sole', 'Japanese Sea Bass', 'Platinum Percula Clownfish', 'lingcod', 'bowfin',
                    'thornyhead', 'Black Sea Bass', 'Warty Sea Cucumber', 'Danio', "Carpenter's Flasher Wrasse",
                    'filefish', 'Blackear Wrasse', 'Milkfish', 'Mango Clownfish', 'cowfish', 'Blue Catfish',
                    'hogfish', 'platypus']

    # Load the pre-trained InceptionV3 model
    model = InceptionV3(weights='imagenet')

    # Function to predict objects in an image
    def predict_objects(image_data):
        # Decode base64 image data
        try:
            image_data_decoded = base64.b64decode(image_data)
        except Exception as e:
            print(f"Error decoding base64 image data: {e}")
            return -1  # Or any other appropriate error code or value

        # Convert to PIL Image
        image = Image.open(BytesIO(image_data_decoded)).resize((299, 299))  # InceptionV3 input size

        # Convert 4-channel images to 3 channels (RGBA to RGB)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Expand dimensions and preprocess input
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        # Make a prediction using the pre-trained model
        predictions = model.predict(image_array)

        # Decode and get the top predicted labels
        decoded_predictions = decode_predictions(predictions, top=10)[0]
        top_labels = [label for (_, label, _) in decoded_predictions]

        predicted_species = 0
        for x in top_labels:
            if x in species_list:
                predicted_species += 1

        return predicted_species

    result = predict_objects(image_data)
    return result

def checkSpecies(model_path, base64_image):
    def predict_with_saved_model(model, image_data):
        # Decode base64 image data
        image_data_decoded = base64.b64decode(image_data)

        # Convert to PIL Image
        img = image.load_img(BytesIO(image_data_decoded), target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class label
        predicted_class = np.argmax(predictions)

        return predicted_class

    def map_index_classes(index_id):
        map_dict = {
            0: 'Bangus', 1: 'Big Head Carp', 2: 'Black Spotted Barb', 3: 'Catfish', 4: 'Climbing Perch',
            5: 'Fourfinger Threadfin', 6: 'Freshwater Eel', 7: 'Glass Perchlet', 8: 'Goby', 9: 'Gold Fish',
            10: 'Gourami', 11: 'Grass Carp', 12: 'Green Spotted Puffer', 13: 'Indian Carp',
            14: 'Indo-Pacific Tarpon', 15: 'Jaguar Gapote', 16: 'Janitor Fish', 17: 'Knifefish',
            18: 'Long-Snouted Pipefish', 19: 'Mosquito Fish', 20: 'Mudfish', 21: 'Mullet', 22: 'Pangasius',
            23: 'Perch', 24: 'Scat Fish', 25: 'Silver Barb', 26: 'Silver Carp', 27: 'Silver Perch', 28: 'Snakehead',
            29: 'Tenpounder', 30: 'Tilapia'
        }

        class_name = map_dict.get(index_id)

        return class_name

    # Load the model
    model = load_model(model_path)

    # Test the model
    predict = predict_with_saved_model(model, base64_image)
    class_name = map_index_classes(predict)

    return class_name

def predict_Model(saved_model_path, base64_image):
    try:
        # Load the saved model
        model = load_model(saved_model_path)

        # Decode base64 image data
        image_data_decoded = base64.b64decode(base64_image)

        # Convert to PIL Image
        img = image.load_img(BytesIO(image_data_decoded), target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make predictions
        prediction = model.predict(img_array)

        # Return prediction
        return 'Fresh' if prediction[0][0] < 0.5 else 'Spoiled'
    except Exception as e:
        # Log any errors
        print("Error predicting:", e)
        return None



# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/user_tab')
def user_tab():
    return render_template('user_tab.html')

@app.route('/register', methods=['GET', 'POST'])
def register(mysql=None):
    if request.method == 'POST':
        # Fetch form data
        userDetails = request.form
        username = userDetails['username']
        password = userDetails['password']

        # Validate form data
        if not username or not password:
            flash('Please fill out all the fields', 'error')
            return redirect(url_for('register'))

        # Create MySQL connection and cursor
        cur = mysql.connection.cursor()

        # Check if the username already exists
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cur.fetchone()

        if existing_user:
            flash('Username already exists. Choose a different one.', 'error')
            return redirect(url_for('register'))

        # Insert user data into the database
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))

        # Commit changes and close cursor
        mysql.connection.commit()
        cur.close()

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Fetch form data
        userDetails = request.form
        username = userDetails['username']
        password = userDetails['password']

        # Create MySQL connection and cursor
        cur = mysql.connection.cursor()

        # Check if the username and password match
        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()

        # Close cursor
        cur.close()

        if user:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid login credentials. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/card1')
def card1():
    return render_template('card1.html')


@app.route('/card2')
def card2():
    # Retrieve data from MongoDB
    fish_data = collection.find()
    # Pass data to the frontend
    return render_template('card2.html', fish_data_result=fish_data)

@app.route('/card3')
def card3():

    # Pass data to the frontend
    return render_template('card3.html')


@app.route('/card4')
def card4():
    # Pass data to the frontend
    return render_template('card4.html')


@app.route('/details/<species_name>')
def details(species_name):
    try:
        # Retrieve data from MongoDB based on species_name
        fish_data = collection.find_one({"fish_name": species_name})

        if fish_data:
            # Pass data to the frontend
            return render_template('details.html', fish_data=fish_data)
        else:
            # Handle case where species_name is not found
            return render_template('error.html', error_message='Fish not found')

    except Exception as e:
        print(f"Error: {e}")

@app.route('/process_data', methods=['GET'])
def process_data():
    # You can perform any necessary data processing or validation here
    
    # Render the test.html template
    return render_template('test.html')


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        print("Result Route Fired...")
        # Handle the POST request
        species_name = request.form.get('species_name')
        image_data = request.form.get('image_data')

        # check freshness
        eye_freshness = predict_Model('W:\\APP - saved\\APP\\media\\sucess_models\\trained_models\\Eyes_Model.h5',image_data)
        print("Eye:",eye_freshness)

        # check gill freshness
        gill_freshness = predict_Model('W:\\APP - saved\\APP\\media\\sucess_models\\trained_models\\Gill_Model.h5',image_data)
        print("Gill:",gill_freshness)

        if eye_freshness and gill_freshness == 'Fresh':
            global result
            result = 'Fresh'
        else:
            result = 'Not Fresh'

        print('Final Result:',result)


        # Fetch the data
        fetch_data = collection.find_one({'fish_name':species_name})

        print("Opening the results web page")
        return render_template('results.html',fish_data=fetch_data,freshness_result=result)
    else:
        # Handle other HTTP methods
        return 'Method Not Allowed', 405

@app.route('/scan', methods=['POST'])
def scan():
    try:
        # Get the image data from the request
        image_data = request.form['image']

        # check if the given image is fish or not
        try:
            check_fish_result = ifFishOrNot(image_data)
            print(check_fish_result)
        except Exception as e:
            print(e)

        if check_fish_result >= 1:
            prediction = 'fish'
            
            try:
            # Replace this with the actual species name obtained from your model
                species_detected = checkSpecies('W:\\APP\\media\\sucess_models\\trained_models\\fishSpeciesPredictionModel.h5',image_data)
                print(species_detected)
            except Exception as e:
                print('Exception',e) 

        else:
            prediction = 'not fish'
            species_detected = 'N/A'

        return jsonify({
            'result': {
                'check_fish_result': prediction,
                'species_detected': species_detected,
                'image_data': image_data  # Add image data to the response
            }
        }), 200


    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def test():
    return render_template('test.html')


