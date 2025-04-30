from imports import *

def generate_user_data(num_users=2000):
    """
    Génère des données utilisateurs fictives avec une cohérence entre les différents champs.
    Assure que la date d'inscription soit avant la date de la dernière connexion d'au moins un mois.
    Insertion aléatoire de None pour les champs non essentiels comme les réseaux sociaux, photo de profil, etc.
    """

    users = []
    for _ in range(num_users):
        # Générer la date d'inscription dans la dernière décennie
        signup_date = fake.date_this_decade()

        # Générer la dernière connexion au moins un mois après la date d'inscription
        last_login = fake.date_this_year()

        # Assurer que la dernière connexion soit au moins un mois après la date d'inscription
        while last_login < signup_date + timedelta(days=30):  # Si la date de dernière connexion est trop proche
            last_login = fake.date_this_year()  # Re-générer la date de la dernière connexion

        # Créer un utilisateur avec des données cohérentes
        user = {
            'user_id': fake.uuid4(),  # Identifiant unique pour chaque utilisateur
            'first_name': fake.first_name(),  # Prénom de l'utilisateur
            'last_name': fake.last_name(),  # Nom de l'utilisateur
            'email': fake.email(),  # Adresse email de l'utilisateur
            'city': fake.city(),  # Ville de l'utilisateur
            'country': fake.country(),  # Pays de l'utilisateur
            'phone_number': fake.phone_number(),  # Numéro de téléphone
            'signup_date': signup_date,  # Date d'inscription (doit être avant la dernière connexion)
            'last_login': last_login,  # Dernière connexion (doit être après la date d'inscription d'au moins un mois)
            'dob': fake.date_of_birth(minimum_age=18, maximum_age=90),  # Date de naissance de l'utilisateur (18 à 90 ans)
            'gender': random.choice(['Male', 'Female', 'Other']),  # Sexe de l'utilisateur
            'profile_picture': None if random.random() < 0.3 else fake.image_url(),  # Photo de profil (30% de chance d'être None)
            'language': random.choice(['English', 'Spanish', 'French', 'German', 'Portuguese', 'Italian', 'Dutch']),  # Langue préférée
            'social_media_handles': {  # Gestion des comptes sociaux (50% de chance de laisser None)
                'facebook': None if random.random() < 0.5 else fake.user_name(),
                'twitter': None if random.random() < 0.5 else fake.user_name(),
                'instagram': None if random.random() < 0.5 else fake.user_name(),
                'linkedin': None if random.random() < 0.5 else fake.user_name(),
            },
            'newsletter_subscribed': random.choice([True, False]),  # Si l'utilisateur est abonné à la newsletter
            'subscription_plan': random.choice(['Free', 'Premium', 'VIP']),  # Plan d'abonnement de l'utilisateur
            'account_status': random.choice(['Active', 'Inactive', 'Suspended']),  # Statut de l'utilisateur
            'referral_source': None if random.random() < 0.7 else random.choice(['Direct', 'Search Engine', 'Social Media', 'Email Campaign', 'Paid Ad']),  # 70% de chance d'avoir None pour la source de référence
            'purchase_history': [  # Historique d'achats simulé
                {
                    'order_id': fake.uuid4(),  # ID de la commande
                    'product': fake.bs(),  # Nom du produit
                    'amount_spent': round(random.uniform(10, 500), 2),  # Montant dépensé pour chaque commande
                    'purchase_date': fake.date_this_year(),  # Date d'achat
                    'status': random.choice(['Shipped', 'Delivered', 'Cancelled', 'Pending'])  # Statut de la commande
                } for _ in range(random.randint(1, 10))  # L'utilisateur peut avoir entre 1 et 10 achats
            ],
            'loyalty_points': random.randint(0, 5000),  # Points de fidélité accumulés
            'last_purchase_date': fake.date_this_year(),  # Date de la dernière commande de l'utilisateur
            'total_spent': round(random.uniform(50, 10000), 2)  # Total dépensé par l'utilisateur
        }
        
        # Ajouter l'utilisateur à la liste
        users.append(user)
    
    return users

def get_products_from_api(num_products=2000):
    """
    Cette fonction simule un appel API pour obtenir des produits.
    Si la connexion à l'API échoue, elle renvoie l'erreur correspondante.
    Simule aussi une faible probabilité d'échecs d'API (404, 429, etc.).
    Insère également des valeurs 'None' dans certains champs pour simuler des données incomplètes.
    """
    
    # Simulation d'une erreur aléatoire avec faible probabilité
    error_probabilities = [404, 429, 500, 502, 503, 504]  # Codes d'erreurs à simuler
    error_chance = random.random()  # Génère un nombre entre 0 et 1
    
    if error_chance < 0.1:  # 10% chance d'avoir une erreur aléatoire
        simulated_error = random.choice(error_probabilities)  # Choisir un code d'erreur
        print(f"API call failed with error: {simulated_error}")
        return {"error": simulated_error}  # Retourner l'erreur simulée, retour immédiat sans produits

    # Simuler une réponse d'API réussie
    print("Simulating API call to get products...")

    # Simulation de la réponse API avec un statut réussi
    simulated_response = {
        'status_code': 200,
        'json': lambda: [{'product_id': fake.uuid4(),
                         'name': fake.bs(),
                         'price': round(random.uniform(5, 500), 2),
                         'category': random.choice(['Electronics', 'Clothing', 'Toys', 'Books', 'Home Appliances', 'Sports', 'Automotive']),
                         'description': fake.text(max_nb_chars=400) if random.random() > 0.1 else None,  # 10% chance d'avoir une description manquante
                         'image_url': fake.image_url() if random.random() > 0.05 else None,  # 5% chance d'avoir une URL d'image manquante
                         'brand': fake.company(),
                         'rating': round(random.uniform(1, 5), 1) if random.random() > 0.2 else None,  # 20% chance d'avoir une note manquante
                         'stock_quantity': random.randint(1, 1000),
                         'shipping_cost': round(random.uniform(5, 50), 2),
                         'weight': round(random.uniform(0.1, 50), 2) if random.random() > 0.15 else None,  # 15% chance d'avoir le poids manquant
                         'color': random.choice(['Red', 'Blue', 'Green', 'Black', 'White', 'Silver', 'Gold']),
                         'size': random.choice(['S', 'M', 'L', 'XL', 'XXL']),
                         'release_date': fake.date_this_year() if random.random() > 0.05 else None,  # 5% chance d'avoir une date de lancement manquante
                         'discount': round(random.uniform(5, 50), 2) if random.random() > 0.1 else None,  # 10% chance d'avoir une remise manquante
                         'manufacturer': fake.company(),
                         'material': random.choice(['Plastic', 'Metal', 'Wood', 'Fabric', 'Leather']),
                         'stock_on_order': random.randint(0, 500) if random.random() > 0.05 else None}  # 5% chance de stock en commande manquant
                        for _ in range(num_products)]
    }

    # On simule la réponse d'une API ici
    response = simulated_response

    if response['status_code'] == 200:
        # La simulation d'API renvoie une liste de produits
        products = response['json']()
        return products  # Retourne les produits
    else:
        # Si la simulation échoue, renvoie le code d'erreur
        return {"error": response['status_code']}

# Fonction pour générer des données de commande
def generate_order_data(num_orders=2000, user_data=None, product_data=None, promo_data=None):
    if user_data is None:
        user_data = []
    if product_data is None:
        product_data = []
    if promo_data is None:
        promo_data = []

    orders = []
    for _ in range(num_orders):
        # Choisir un utilisateur et un produit aléatoirement
        user = random.choice(user_data)
        product = random.choice(product_data)
        
        # Calcul du total prix basé sur la quantité et le prix du produit
        quantity = random.randint(1, 5)  # Quantité achetée entre 1 et 5
        total_price = round(quantity * product['price'], 2)  # Prix total basé sur la quantité

        # Appliquer une promotion aléatoire si présente
        promo = None
        if random.random() < 0.2 and promo_data:  # 20% de chance d'une promotion
            promo = random.choice(promo_data)
            discount_amount = (promo['discount'] / 100) * total_price  # Calcul du montant de remise
            total_price = round(total_price - discount_amount, 2)  # Appliquer la remise

        # Date de la commande
        order_date = fake.date_this_year()

        # Estimer un délai réaliste pour la livraison en fonction du statut
        if random.random() < 0.5:
            shipping_date = fake.date_between(start_date=order_date, end_date=order_date.replace(year=order_date.year + 1))
            status = 'Shipped'
        else:
            shipping_date = None
            status = 'Processing'
        
        # Estimer un délai de livraison en fonction du statut
        if status == 'Shipped':
            delivery_date = fake.date_between(start_date=shipping_date, end_date=shipping_date.replace(month=shipping_date.month + 2))
        else:
            delivery_date = None
        
        # Créer la commande
        order = {
            'order_id': fake.uuid4(),  # Identifiant unique pour chaque commande
            'user_id': user['user_id'],  # Identifiant de l'utilisateur ayant passé la commande
            'product_id': product['product_id'],  # Identifiant du produit commandé
            'quantity': quantity,  # Quantité achetée
            'unit_price': product['price'],  # Prix unitaire du produit
            'total_price': total_price,  # Prix total de la commande
            'order_date': order_date,  # Date de la commande
            'shipping_date': shipping_date,  # Date d'expédition (si applicable)
            'delivery_date': delivery_date,  # Date de livraison (si applicable)
            'status': status,  # Statut de la commande
            'promo_code': promo['promo_code'] if promo else None,  # Code promo appliqué (le cas échéant)
            'discount_applied': promo['discount'] if promo else None,  # Remise appliquée
            'delivery_address': fake.address(),  # Adresse de livraison aléatoire
            'shipping_method': random.choice(['Standard', 'Expedited', 'Next-day']),  # Méthode d'expédition
            'tracking_number': fake.uuid4() if status == 'Shipped' else None  # Numéro de suivi si expédié
        }

        # Ajouter la commande à la liste des commandes
        orders.append(order)

    return orders


# Simulation d'un appel API pour obtenir des interactions sur les réseaux sociaux
def get_social_media_data_from_api(num_interactions=2000, user_data=None):
    if user_data is None:
        user_data = []

    # Exemple d'appel à une API externe pour récupérer les interactions
    response = requests.get("https://api.example.com/social_interactions")  # Remplacer par l'URL réelle de l'API
    if response.status_code == 200:
        interactions = response.json()  # L'API renvoie des données sur les interactions
    else:
        # En cas d'échec de l'API, générer des interactions aléatoires
        interactions = []
        for _ in range(num_interactions):
            interaction = {
                'user_id': random.choice(user_data)['user_id'],  # Choix aléatoire d'un utilisateur
                'interaction_type': random.choice(['Mention', 'Share', 'Comment']),  # Type d'interaction
                'content': fake.text(max_nb_chars=50),  # Contenu de l'interaction
                'platform': random.choice(['Twitter', 'Instagram', 'Facebook', 'TikTok']),  # Plateforme
                'date': fake.date_this_year()  # Date de l'interaction
            }
            interactions.append(interaction)
    return interactions

# Fonction pour générer des données de promotions
def generate_promotion_data(num_promotions=2000):
    promotions = []
    for _ in range(num_promotions):
        promo = {
            'promo_code': fake.bothify(text='??###'),  # Code promo aléatoire
            'discount': round(random.uniform(5, 50), 2),  # Remise entre 5% et 50%
            'start_date': fake.date_this_year(),  # Date de début de la promotion
            'end_date': fake.date_this_year(),  # Date de fin de la promotion
        }
        promotions.append(promo)
    return promotions

# Simulation d'un appel API pour obtenir des données de stock via API
def get_stock_data_from_api(num_products=2000):
    # Simulation d'un appel API pour obtenir des informations de stock produit
    response = requests.get("https://api.example.com/stock_data")  # Remplacer par l'URL réelle de l'API
    if response.status_code == 200:
        stock_data = response.json()  # L'API renvoie les données de stock
    else:
        # En cas d'échec de l'API, générer des données de stock aléatoires
        stock_data = []
        for _ in range(num_products):
            stock = {
                'product_id': fake.uuid4(),  # Identifiant unique pour chaque produit
                'stock_quantity': random.randint(0, 1000),  # Quantité de stock aléatoire entre 0 et 1000
            }
            stock_data.append(stock)
    return stock_data

# Fonction pour simuler l'écriture des données dans un fichier CSV
def save_to_csv(data, filename):
    keys = data[0].keys()  # Extraire les clés pour les colonnes CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"Data saved to {filename}")














