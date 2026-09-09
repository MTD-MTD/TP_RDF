\# Classification de textures par LBP et SVM



Projet du \*\*groupe 2\*\* réalisé dans le cadre du cours de \*\*Reconnaissance de forme\*\*, en Master 1, semestre 2, à l’Université de Kinshasa.



\## Objectif



Reconnaître des catégories de défauts de surface de l’acier à partir d’images étiquetées.



Le système associe :



\- \*\*LBP — Local Binary Pattern\*\* : extraction des caractéristiques de texture ;

\- \*\*SVM — Support Vector Machine\*\* : classification supervisée à noyau RBF.



Le projet utilise exclusivement des méthodes classiques, sans Deep Learning.



\## Méthode



1\. Inventaire des images et des catégories.

2\. Conversion en niveaux de gris.

3\. Extraction d’un histogramme LBP normalisé.

4\. Suppression des doublons exacts.

5\. Séparation en apprentissage, validation et test.

6\. Standardisation des caractéristiques et entraînement du SVM.

7\. Sélection des paramètres sur la validation.

8\. Évaluation sur le test et démonstration.



\## Données



L’archive utilisée, nommée \*\*NEU-CLS-64\*\*, contient initialement \*\*7 226 images\*\*, réparties dans neuf catégories.



| Code | Catégorie | Images initiales |

|---|---|---:|

| cr | Fissures en réseau | 1 210 |

| gg | Rainures et entailles | 296 |

| in | Inclusions | 775 |

| pa | Plaques et taches étendues | 1 148 |

| ps | Surface piquée | 797 |

| rp | Signification exacte à confirmer | 200 |

| rs | Calamine incrustée | 1 589 |

| sc | Rayures | 773 |

| sp | Taches ponctuelles | 438 |



Après suppression de \*\*7 doublons exacts\*\*, les \*\*7 219 images\*\* restantes sont réparties ainsi :



| Ensemble | Images | Utilisation |

|---|---:|---|

| Apprentissage | 4 620 | Ajuster le normaliseur et le SVM |

| Validation | 1 155 | Choisir les paramètres |

| Test | 1 444 | Évaluer les performances |



Le notebook copie les images dans trois dossiers distincts : `train`, `validation` et `test`. Les originaux sont conservés.



Les effectifs ci-dessus correspondent à notre archive ; ils ne doivent pas être confondus avec ceux de la version originale de NEU à six catégories.



\## Configuration



\### Descripteur LBP



\- Voisins : `P = 8`

\- Rayon : `R = 1`

\- Méthode : `uniform`

\- Histogramme : 10 composantes normalisées

\- Bordure exclue : 1 pixel

\- Aucun redimensionnement dans le traitement



\### Classificateur



Le pipeline associe `StandardScaler` et `SVC`.



Les 24 configurations évaluées combinent :



\- `C` : `0.1`, `1`, `10`, `100`

\- `gamma` : `scale`, `0.01`, `1`

\- `class\_weight` : `None`, `balanced`



La sélection utilise le \*\*F1-macro sur une validation fixe\*\*.



Configuration retenue :



```python

SVC(kernel="rbf", C=100, gamma="scale", class\_weight=None)

```



Le normaliseur et le SVM final sont ajustés uniquement sur l’apprentissage.



\## Résultats



| Indicateur | Valeur |

|---|---:|

| F1-macro sur validation | 0,7307 |

| Accuracy sur test | 81,30 % |

| Exactitude équilibrée sur test | 69,82 % |

| F1-macro sur test | 0,7020 |

| Images correctement classées | 1 174 / 1 444 |



Les performances varient selon les catégories :



\- `rs` : \*\*308 images reconnues sur 318\*\* ;

\- `cr` : \*\*228 images reconnues sur 242\*\* ;

\- `rp` : \*\*3 images reconnues sur 39\*\*.



Les 81,30 % de bonnes prédictions ne signifient donc pas que toutes les catégories sont également bien reconnues.



\## Exécution



\### Environnement



Le projet a été exécuté sous Windows avec Anaconda, dans l’environnement `unikin\_ml`.



Bibliothèques utilisées :



\- NumPy

\- SciPy

\- Pillow

\- scikit-image

\- scikit-learn

\- pandas

\- Matplotlib

\- joblib

\- Jupyter / IPython

\- tkinter pour la sélection d’image



\### Lancer le notebook



1\. Ouvrir le notebook SVM final dans Jupyter.

2\. Sélectionner le noyau `unikin\_ml`.

3\. Adapter la variable `DOSSIER` au chemin local du dataset :



```python

DOSSIER = Path(r"C:\\Users\\user\\Desktop\\RDF\\NEU-CLS-64")

```



4\. Exécuter les cellules dans l’ordre.

5\. Attendre la fin de la sélection des paramètres.

6\. Consulter les scores et la matrice de confusion.

7\. Exécuter la cellule de sélection d’image pour la démonstration.



Les fichiers d’origine doivent être organisés selon le modèle `categorie/image.jpg`, par exemple `cr/1.jpg`.



\## Démonstration



La démonstration :



\- accepte uniquement les images du dossier `test` de l’exécution ;

\- vérifie que l’image n’a pas été modifiée ;

\- affiche sa catégorie réelle et sa catégorie prédite ;

\- indique si la prédiction est correcte.



Les noms français sont utilisés pour l’affichage, tandis que les codes restent les étiquettes internes.



\## Sauvegardes



L’étape d’export enregistre notamment :



\- `modele\_lbp\_svm.joblib` : pipeline entraîné et paramètres LBP ;

\- `donnees\_experience.npz` : caractéristiques, étiquettes, chemins et séparations ;

\- `validation\_fixe.csv` : comparaison des configurations ;

\- `predictions\_test.csv` : prédictions individuelles ;

\- `matrice\_confusion.csv` : répartition des prédictions ;

\- `resultats.json` : scores, paramètres et versions enregistrées.



\## Limites



\- Les catégories sont déséquilibrées.

\- Un histogramme LBP global ne conserve pas la position des motifs.

\- Une seule configuration LBP et une seule séparation de validation sont étudiées.

\- Les doublons exacts sont supprimés, mais les quasi-doublons et les découpes apparentées ne sont pas entièrement contrôlés.

\- Le test avait déjà été consulté lors d’une exploration K-means : il ne constitue pas une nouvelle évaluation indépendante.

\- La signification exacte du code `rp` reste à confirmer.

\- Les résultats ne démontrent pas les performances sur d’autres conditions d’acquisition.






\## Références



\- \[Base de données NEU](https://faculty.neu.edu.cn/songkechen/zh\_CN/zdylm/263270/list/)

\- \[Documentation LBP — scikit-image](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local\_binary\_pattern)

\- \[Documentation SVM — scikit-learn](https://scikit-learn.org/stable/modules/svm.html)

\- \[Validation des modèles — scikit-learn](https://scikit-learn.org/stable/modules/cross\_validation.html)

