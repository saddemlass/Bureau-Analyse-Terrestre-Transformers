# Bureau d'Analyse Terrestre - La salle de lecture

## Phase 0 - Refaire les calculs du disparu

### Protocole

Le fichier source est telecharge automatiquement depuis la source officielle si `releves_klaxo3.csv` n'existe pas deja. Le dataset local n'est pas versionne.

Le chargement compte d'abord les lignes physiques du CSV. Ensuite, il charge les lignes a 11 champs, qui correspondent aux colonnes attendues. Les lignes avec trop de champs sont reparees quand le probleme vient de virgules non protegees dans `comments`. Les autres lignes malformees sont comptees a part. On ne les perd donc pas silencieusement.

Bilan du chargement :

- lignes physiques : 88875
- lignes chargees : 88875
- lignes reparees : 196
- lignes malformees non chargees : 0

### Choix de date

J'utilise `datetime` pour les calculs temporels. C'est la date de l'observation. `date_posted` est seulement la date de publication dans la base. Pour savoir quand les temoins disent avoir vu quelque chose dans le ciel, `datetime` est donc le bon champ.

Les heures `24:00` sont converties au jour suivant a `00:00`. Je ne supprime pas ces lignes, car elles portent une information temporelle recuperable.

### Periode testee

J'ai teste les donnees completes :

- releves : 88875
- date min : 1906-11-11
- date max : 2014-05-08
- jours calendaires couverts : 39261
- moyenne : 2.26 releves par jour

Les chiffres du dossier du predecesseur correspondent a la periode 1990-2014. C'est donc cette periode que j'utilise pour reproduire son dossier, sans forcer artificiellement les resultats.

Periode retenue :

- date min : 1990-01-01
- date max : 2014-05-08
- jours calendaires couverts : 8894
- releves retenus : 81678
- moyenne : 9.18 releves par jour

### Chiffres reproduits

- moyenne par jour : 9.18
- samedi : 17.7 %
- lundi : 12.6 %
- juillet : 11.3 %
- fevrier : 6.2 %

Pour le 4 juillet, le chiffre 51 ne correspond pas a un total et ne correspond pas a une journee particuliere. Il correspond a une moyenne calendaire arrondie : 1220 releves tombent un 4 juillet, sur 24 dates calendaires du 4 juillet couvertes par la periode. Cela donne 50.83, donc 51 releves arrondis.

La moyenne limitee aux 4 juillet qui ont au moins un releve est 53.04 releves, sur 23 dates observees. Le 4 juillet le plus charge est 2010-07-04, avec 206 releves.

### Chiffres supplementaires

- maximum quotidien toutes dates confondues : 206 releves
- date du maximum : 2010-07-04
- 4 juillet le plus charge : 2010-07-04, 206 releves
- rang du 4 juillet le plus charge parmi toutes les journees individuelles : 1

Ici, le rang du 4 juillet signifie : on classe chaque jour calendrier individuel par nombre de releves, puis on regarde la position du 4 juillet qui a le plus de releves.

### Top 10 des journees

| date | nombre_releves | jour_semaine | mois |
| --- | --- | --- | --- |
| 2010-07-04 | 206 | Sunday | 7 |
| 1999-11-16 | 195 | Tuesday | 11 |
| 2012-07-04 | 192 | Wednesday | 7 |
| 2013-07-04 | 180 | Thursday | 7 |
| 2011-07-04 | 155 | Monday | 7 |
| 2009-09-19 | 129 | Saturday | 9 |
| 2014-01-01 | 99 | Wednesday | 1 |
| 2013-12-31 | 96 | Tuesday | 12 |
| 2004-10-31 | 94 | Sunday | 10 |
| 2009-07-04 | 88 | Saturday | 7 |

Le tableau est aussi enregistre dans `outputs/phase0_top10_journees.csv`.

### Volume annuel

- 1990 : 295
- 1991 : 261
- 1992 : 291
- 1993 : 361
- 1994 : 476
- 1995 : 1441
- 1996 : 992
- 1997 : 1406
- 1998 : 1995
- 1999 : 3126
- 2000 : 3057
- 2001 : 3494
- 2002 : 3648
- 2003 : 4393
- 2004 : 4711
- 2005 : 4478
- 2006 : 4121
- 2007 : 4688
- 2008 : 5247
- 2009 : 4941
- 2010 : 4695
- 2011 : 5530
- 2012 : 7946
- 2013 : 7608
- 2014 : 2477

Verdict sur la phrase "le volume annuel croit continument jusqu'a la fin" : FAUX.

Des baisses existent : 1991 (295 vers 261), 1996 (1441 vers 992), 2000 (3126 vers 3057), 2005 (4711 vers 4478), 2006 (4478 vers 4121), 2009 (5247 vers 4941), 2010 (4941 vers 4695), 2013 (7946 vers 7608), 2014 (7608 vers 2477).

La courbe est enregistree dans `outputs/phase0_volume_annuel.png`.

## Phase 1 - Le chiffre etait vrai, la flotte est perdue

### Partie 1 - Ce que dit vraiment le chiffre du 4 juillet

Le chiffre du 4 juillet mesure un volume de signalements. Il dit qu'un certain jour, beaucoup de lignes ont ete enregistrees dans la base.

Il ne prouve pas a lui seul que les gens ne preteront pas attention a quelque chose. Il ne prouve pas non plus que les observations sont banales, que les temoins sont plus credules, ou que tous les evenements observes sont de meme nature.

Plusieurs explications peuvent produire le meme pic. Le 4 juillet, il y a des feux d'artifice. Plus de personnes sont dehors le soir. Il peut aussi y avoir plus de lumieres inhabituelles dans le ciel, ou plus de confusions avec des evenements lumineux connus. Les donnees permettent de constater le pic de signalements. Elles ne suffisent pas a expliquer toutes ses causes.

### Partie 2 - Trois releves reels

### Releve 1
- datetime : 2/9/1990 02:30
- lieu : plattsburg, ny, 
- shape : light
- comments : A bright yellow light follwing my car on the highway...and then just dissapeared.

### Releve 2
- datetime : 1/1/1990 18:00
- lieu : delhi (india), , 
- shape : triangle
- comments : Saw triangle shape plane with disco lights all over its edges standing over my house

### Releve 3
- datetime : 3/1/1990 21:00
- lieu : williston/ archer, fl, 
- shape : fireball
- comments : A blue&#44 firey ball&#44 about 20 feet wide and 150 feet in the air&#44 flies over head at a slow rate making no sound and lighting up the goun

Ces trois lignes sont comptees de la meme maniere dans les statistiques. Pourtant, elles ne racontent pas la meme chose. Une ligne peut etre courte, une autre peut decrire une lumiere, une autre peut raconter une observation plus precise ou plus confuse. Le comptage mesure d'abord un volume de temoignages.

### Partie 3 - Nouvelle tache ML

ENTREE : le texte `comments` ecrit par le temoin.

SORTIE : la forme `shape` observee.

"A partir du temoignage ecrit par un temoin, predire la forme de l'objet qu'il decrit."
