
// Deux Types de loop, If et Switch (probablement égal au FOR ou au Case du SQL, on verra)

//Égalité de valeur	==	//
let prenom = "Claude"

// Cette expression nous renverra la valeur "true"
(prenom=="Claude")

// Égalité de valeur et de type	===	let age = 18

//Cette expression nous renverra "false" car 18 est un //type number alors que "18" est un type string
// Commentaire perso : étrange, car lors de l'input du précédent exercice, en l'absence de conversion de l'input, js avait considéré 5 et 7 comme des string et m'avait rendu 57... Et maintenant il dit que 8 est considéré comme int sans conversion. A demander

(age==="18")

// Plus grand, Plus petit que	>, < //
let age=36

(age>36) //age strictement supérieur à 36 ? false

// Plus grand ou égal & Plus petit ou égal	>=, <=	
let age2=36

(age2>=36) //age plus grand ou égal à 36 ? true

// Différent de !=	//
let age3=36

(age3 !=36) //age différent de 36 ? false


////////////// LES OPERATEURS LOGIQUES ////////////////

// ET	&&	//

age<18 && age>=12
//La variable age doit être inférieur à 18 ET EGALEMENT supérieur ou égale 12 pour que l'expression renvoie true

//Com perso : Pq pas de parenthèse ? A demander

OU	||	age>=18 || isMineurAnticipe

//L’age doit être supérieur ou égal à 18 OU la variable
//isMineurAnticipe doit valoir true pour que l'expression renvoie true

// NON	!	

!majeur //Sera traduit par "N'est pas majeur" ou "est NON majeur"

//L'expression renvoie true si majeur==false
//L'expression renvoie false si majeur==true

/// STRUCTURE ET SYNTAXE DU IF ///

if ( expression1 && expression2 || expression3)
    {
    
              //Code à exécuter si la condition est vraie
    
    }

// Exemple

var nombre = -42;

if (nombre > 0) {

  console.log("Le nombre est positif");

} else {

  console.log("Le nombre est négatif ou nul");

}



// ELSE IF  est l'équivalent du elif en Python


if ( expression1 && expression2 || expression3)
    {
    
              //Code à exécuter si la condition est vraie
    
    } else if ( expression1 && expression2 || expression3){
    
            //Code à exécuter si la deuxieme condition est vraie
    
    } else {
         
            //Code à exécuter si aucune condition n'est vraie
    
    }


///// SWITCH .... CASE

var jour = "lundi";

switch (jour) {
  case "lundi":
    console.log("C'est le lundi");
    break;
  case "mardi":
    console.log("C'est le mardi");
    break;
  case "mercredi":
    console.log("C'est le mercredi");
    break;
  default:
  // Le default rappelle le default de SSIS. est-ce qu'il y a un lien ? A demander...  
  console.log("Je ne sais pas quel jour c'est");
    break;
}

//// Cumuler plusieurs case consécutivement dans une proposition:

var nombre = 3;

switch (nombre) {
  case 1:
  case 2:
  case 3:
    console.log("Le nombre est inférieur ou égal à 3");
    break;
  case 4:
    console.log("Le nombre est égal à 4");
    break;
  default:
    console.log("Je ne sais pas combien est le nombre");
    break;
}

/// Les tableau /// 

//Création du tableau "monTableau"
let monTableau = [valeur1, valeur2, valeur3]
// Mais c'est comme une Liste Python... On va voir

// Par exemple, si vous voulez stocker une liste de noms, 
// vous pouvez les mettre tous dans un tableau :

let noms = ["Alice", "Bob", "Charlie"];
// c'est littéralement une liste! Pourquoi ils appellent ca un tableau ??

// .length == python len(x)
console.log(noms.length)

//// PUSH METHOD = append method

let fruits = ["banane", "orange"];

//On va ajouter la valeur "pomme" à la fin du tableau
fruits.push("pomme");

// Affiche ["banane", "orange", "pomme"] dans la console
console.log(fruits);

//// POP METHOD : Delete the last item

//On supprime la dernière entrée du tableau
fruits.pop();

 // Affiche ["banane", "orange"] dans la console
console.log(fruits);


//// SHIFT METHOD : Delete the first item

//On supprime la première entrée du tableau
fruits.shift();

 // Affiche ["orange", "pomme"] dans la console
console.log(fruits);

//// UNSHIFT METHOD : ajoute un ou plusieurs éléments au début du tableau

//Ajouter un élément au début du tableau
fruits.unshift("banane");

 // Affiche ["banane", "orange", "pomme"] dans la console
console.log(fruits);


///// 

// L'initialisation : 1ère partie de la structure, 
// Prépare une variable compteur pour garder un oeil sur la progression de la structure.

// 2 Condition de répétition : 
// Condition qui, tant qu'elle est vraie, permet la répétition du code. 
// Généralement basée sur la valeur de la variable compteur.

// 3 L'incrémentation : 
// Exécutée à la fin de chaque tour de boucle pour mettre à jour la variable compteur
// Permet une invalidation de la condition de répétition et donc un arrêt de la boucle.

// Par exemple, pour afficher chaque élément du tableau ["banane", "pomme", "orange", "kiwi"], on peut écrire une boucle for comme suit : 

let fruits2 = ["banane", "pomme", "orange", "kiwi"];

for(let i = 0; i < fruits2.length; i++) {

   console.log(fruits2[i]);
}


//Boucle While

let fruits3 = ["banane", "pomme", "orange", "kiwi"];
let i = 0;

while (i < fruits3.length) {

  console.log(fruits3[i]);
  i++;

}

// DO While

do {

  console.log(fruits3[i]);
  i++;

} while (i < fruits3.length);


