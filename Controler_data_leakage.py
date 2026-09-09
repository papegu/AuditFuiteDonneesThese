from pathlib import Path

base = Path("/mnt/data/annexe_shi_spi_leakage_free")
base.mkdir(parents=True, exist_ok=True)

tex = r"""% ==============================================================================
% ANNEXE — Corrections méthodologiques des pipelines SHI et SPI
% Document destiné au dépôt GitHub avec les scripts Python corrigés.
%
% Compilation recommandée : XeLaTeX ou LuaLaTeX
% Dépendances : book, babel, geometry, listings, xcolor, float, booktabs,
%               amsmath, amssymb, enumitem, hyperref
%
% Arborescence recommandée :
%   annexe_shi_spi_leakage_free.tex
%   code/
%       shi_lucas2018_leakage_free.py
%       spi_satellite_leakage_free.py
% ==============================================================================

\documentclass[11pt,a4paper]{report}

\usepackage[french]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{float}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{hyperref}

\hypersetup{
    colorlinks=true,
    linkcolor=black,
    urlcolor=blue,
    citecolor=black
}

\lstdefinestyle{pythonstyle}{
    language=Python,
    basicstyle=\ttfamily\footnotesize,
    keywordstyle=\color{blue}\bfseries,
    commentstyle=\color{gray}\itshape,
    stringstyle=\color{teal},
    numbers=left,
    numberstyle=\tiny\color{gray},
    stepnumber=1,
    numbersep=8pt,
    frame=single,
    breaklines=true,
    breakatwhitespace=false,
    showstringspaces=false,
    tabsize=4,
    captionpos=b,
    columns=fullflexible,
    keepspaces=true
}

\title{Corrections méthodologiques des pipelines\\
SHI et SPI}
\author{Pape El Hadji Abdoulaye Gueye}
\date{\today}

\begin{document}

\maketitle

\tableofcontents

% ==============================================================================
\chapter{Correction du pipeline de construction et de prédiction du
\textit{Soil Health Index} (SHI)}
\label{annexe:shi-leakage-free}

Cette annexe documente les corrections méthodologiques apportées au pipeline
de construction et de prédiction du \textit{Soil Health Index} (SHI) à partir
du jeu de données LUCAS 2018, en réponse à plusieurs sources de fuite de
données (\textit{data leakage}) identifiées dans une version antérieure du
script.

\section{Problèmes identifiés dans la version initiale}

La version initiale du pipeline présentait quatre sources potentielles de
fuite de données (\textit{data leakage}) :

\begin{enumerate}
    \item l'imputation des valeurs manquantes par la médiane était calculée
    sur l'ensemble du jeu de données \emph{avant} la séparation entre les
    ensembles d'entraînement et de test ;

    \item les pondérations du SHI étaient optimisées par recherche exhaustive
    (\textit{grid search}) en utilisant le score $R^2$ calculé sur le jeu de
    test. Cette procédure revenait à sélectionner la définition de la variable
    cible en fonction des données destinées à l'évaluation finale ;

    \item certaines composantes biologiques ayant servi à construire le SHI,
    notamment \textit{Cmic}, \textit{Xylosidase} et
    \textit{Basal\_respiration}, étaient réintroduites comme variables
    prédictives, créant ainsi une fuite conceptuelle directe entre la cible
    et les prédicteurs ;

    \item le choix du meilleur modèle parmi les algorithmes évalués reposait
    sur les performances mesurées sur le jeu de test plutôt que sur une
    validation croisée réalisée exclusivement sur l'ensemble d'entraînement.
\end{enumerate}

\section{Corrections apportées}

Le pipeline a été intégralement révisé afin de garantir une séparation stricte
entre les données d'entraînement et les données d'évaluation, conformément aux
bonnes pratiques de validation en apprentissage automatique appliqué aux
sciences du sol.

Les principales corrections sont les suivantes :

\begin{itemize}
    \item les pondérations du SHI,
    \[
    \mathrm{SHI}
    =
    0.4\,\mathrm{Cmic}
    +0.1\,\mathrm{Basal\_respiration}
    +0.1\,\mathrm{Xylosidase}
    +0.1\,\mathrm{Beta\_glucosidase}
    +0.1\,\mathrm{Acid\_phosphatase}
    +0.2\,\mathrm{N\_acetylglucosaminidase},
    \]
    ont été fixées \textit{a priori} sur la base de la définition retenue
    pour l'indice et ne sont plus ré-optimisées à partir du jeu de test ;

    \item l'imputation des valeurs manquantes, réalisée avec
    \texttt{SimpleImputer} selon une stratégie par médiane, et la
    standardisation, réalisée avec \texttt{StandardScaler}, sont ajustées
    (\texttt{fit}) exclusivement sur l'ensemble d'entraînement, puis
    appliquées (\texttt{transform}) à l'ensemble de test ;

    \item les variables prédictives, à savoir \textit{WHC},
    \textit{Mean\_width\_diameter}, \textit{Water\_stable\_aggregates} et
    \textit{MIRR}, sont vérifiées comme étant disjointes des composantes
    biologiques utilisées pour construire le SHI au moyen d'une assertion
    automatique :
\begin{verbatim}
assert set(shi_components).isdisjoint(set(predictor_variables))
\end{verbatim}

    \item la sélection du meilleur modèle parmi Random Forest, XGBoost,
    CatBoost et LightGBM est effectuée au moyen d'une validation croisée à
    5 plis (\textit{5-fold cross-validation}) réalisée exclusivement sur
    l'ensemble d'entraînement. Le jeu de test n'intervient qu'une seule fois,
    lors de l'évaluation finale du modèle retenu ;

    \item l'analyse d'explicabilité SHAP est réalisée après le verrouillage
    des modèles finaux et n'est pas utilisée pour modifier ultérieurement
    les modèles, les hyperparamètres ou les variables prédictives ;

    \item un module d'audit automatique (\textit{leakage audit}) vérifie,
    à l'issue de l'exécution, le respect des principaux critères d'absence de
    fuite de données.
\end{itemize}

\section{Résultats obtenus après correction}

Sur les 881 observations exploitables du jeu de données LUCAS 2018, dont
704 observations utilisées pour l'entraînement et 177 pour le test, selon
une séparation aléatoire de 80/20 avec \texttt{random\_state=42}, le modèle
CatBoost obtient les meilleures performances sur le jeu de test indépendant,
avec un coefficient de détermination de $R^2=0.850$, une MAE de $51.2$ et une
RMSE de $80.0$.

Ces performances sont cohérentes avec celles obtenues lors de la validation
croisée sur l'ensemble d'entraînement, pour laquelle CatBoost atteint
$R^2_{\mathrm{CV}}=0.819$. Cette cohérence entre la validation croisée et
l'évaluation indépendante renforce la crédibilité des résultats obtenus
après correction du pipeline.

L'audit automatisé de fuite de données confirme le respect des contraintes
méthodologiques définies précédemment et retourne le statut
\texttt{LEAKAGE AUDIT: PASSED}.

\section{Code source complet}
\label{annexe:shi-leakage-free-code}

Le code source complet du pipeline corrigé est fourni séparément dans le
répertoire \texttt{code/}. Le script recommandé est
\texttt{shi\_lucas2018\_leakage\_free.py}.

Pour l'inclure directement dans une version destinée à la thèse, utiliser :

\begin{verbatim}
\lstinputlisting[style=pythonstyle,
caption={Pipeline SHI corrigé et sans fuite de données}]
{code/shi_lucas2018_leakage_free.py}
\end{verbatim}

Le fichier Python doit donc être placé dans le sous-dossier
\texttt{code/} du projet LaTeX. Cette organisation permet de conserver
séparément le texte méthodologique et le code exécutable.

\section{Référencement du dépôt de code}

Le dépôt GitHub associé doit être indiqué ici lorsque son URL définitive
est disponible :

\begin{center}
\texttt{[URL DU DÉPÔT GITHUB À RENSEIGNER]}
\end{center}

% ==============================================================================
\chapter{Pipeline de comparaison BiLSTM--CatBoost et test de
Diebold--Mariano}
\label{annexe:bilstm-catboost-dm}

Cette annexe présente le pipeline expérimental utilisé pour comparer les
performances du modèle BiLSTM et du modèle CatBoost dans un cadre de
validation temporelle sans fuite de données. Le protocole repose sur une
validation croisée \textit{walk-forward} utilisant
\texttt{TimeSeriesSplit}, de manière à garantir que les observations
utilisées pour l'entraînement précèdent strictement les observations
utilisées pour la validation.

Une attention particulière est portée à la séparation entre les observations
réelles et les observations synthétiques. Les observations synthétiques
peuvent être utilisées pendant l'entraînement afin de densifier les séries
temporelles, mais elles sont systématiquement exclues des métriques finales
et du test de Diebold--Mariano. Ainsi, les performances rapportées pour la
comparaison des modèles reposent exclusivement sur des observations réelles.

Pour chaque pli temporel, les variables de transformation sont ajustées
exclusivement sur les données d'entraînement. Les paramètres des
\texttt{MinMaxScaler} utilisés pour les variables dynamiques, les variables
statiques et la variable cible sont donc estimés sur le sous-ensemble
d'entraînement, puis appliqués au sous-ensemble de validation. De même, la
variable \texttt{Previous\_Yield} est reconstruite indépendamment dans
chaque pli afin d'éviter l'utilisation d'informations futures.

Le modèle BiLSTM exploite une séquence temporelle de longueur $T=3$ et
combine les variables dynamiques avec les variables statiques. CatBoost
reçoit les mêmes séquences temporelles aplaties et les mêmes variables
statiques, ce qui permet une comparaison fondée sur les mêmes observations
et les mêmes informations disponibles.

Les performances sont évaluées à l'aide du RMSE, du MAPE et du coefficient
de détermination $R^2$. Enfin, les erreurs de prédiction hors échantillon
obtenues sur les observations réelles de l'ensemble des plis sont
concaténées afin d'effectuer le test de Diebold--Mariano. Ce test permet
d'évaluer si la différence de précision prédictive entre le BiLSTM et
CatBoost est statistiquement significative.

\section{Code source}

Le script Python correspondant doit être placé dans le répertoire
\texttt{code/}, par exemple sous le nom
\texttt{bilstm\_catboost\_dm\_leakage\_free.py}.

% ==============================================================================
\chapter{Correction d'une fuite de données dans la Contribution 3
(prédiction du SPI)}
\label{annexe:fuite-spi}

\section{Contexte et diagnostic du problème}

Une relecture méthodologique a posteriori de la Contribution 3 a permis
d'identifier une fuite de données (\textit{data leakage}) affectant
l'ensemble des modèles présentés initialement (Random Forest, XGBoost,
SVR, ELM). Cette annexe documente le diagnostic, la correction apportée
et les résultats obtenus après correction, dans un souci de transparence
méthodologique.

Le Standardized Precipitation Index (SPI) est, dans le pipeline considéré,
une transformation affine de la précipitation cumulée sur la période
étudiée :
\[
SPI=\frac{X-\mu}{\sigma},
\]
où $X$ est la précipitation totale observée
(\textit{TotalPrecip} dans le jeu de données), et $\mu$, $\sigma$ sont
respectivement la moyenne et l'écart-type historiques de cette
précipitation.

Cette relation a été vérifiée empiriquement sur le jeu de données utilisé :
la corrélation entre \textit{TotalPrecip} et le SPI est de $1.000000$, et
l'écart absolu entre le SPI observé et le score-$z$ de
\textit{TotalPrecip} est de l'ordre de $10^{-15}$, ce qui correspond au
bruit de calcul en virgule flottante. Cette vérification confirme qu'il
s'agit, dans ce jeu de données, d'une identité mathématique et non d'une
simple corrélation forte.

Or, dans la chaîne de modélisation initiale, \textit{TotalPrecip} figurait
parmi les variables explicatives utilisées pour prédire le SPI. Cela revient
à fournir au modèle une transformation déterministe de la cible elle-même.
Les algorithmes peuvent alors redécouvrir la relation
$(X-\mu)/\sigma$ au lieu d'apprendre une relation prédictive indépendante.

Ce phénomène explique les performances quasi parfaites rapportées
initialement : $R^2=0.9991$ pour Random Forest, $R^2=0.9984$ pour XGBoost
et $R^2=0.9971$ pour SVR, ainsi que la domination écrasante de
\textit{TotalPrecip} dans l'analyse SHAP initiale.

Une fuite secondaire, de moindre ampleur, a également été identifiée dans
certaines variantes du pipeline : la normalisation
(\texttt{MinMaxScaler}) était ajustée (\texttt{fit\_transform}) sur
l'ensemble des données \textbf{avant} la séparation train/test, permettant
aux statistiques (min/max) de l'ensemble de test de fuiter dans
l'entraînement.

\section{Correction méthodologique apportée}

Les corrections suivantes ont été appliquées :

\begin{enumerate}
    \item \textbf{Suppression de \textit{TotalPrecip} des variables
    explicatives.} Seules les variables dérivées de l'imagerie satellite
    (composantes principales PC1 à PC8, issues des bandes spectrales B2,
    B3, B4, B8 et des indices NDVI, NDWI, MSAVI, EVI) sont conservées comme
    prédicteurs. Ce choix est cohérent avec l'objectif scientifique réel de
    la contribution : estimer le risque climatique à partir de
    l'observation satellite seule, dans les zones où la mesure directe de
    précipitation au sol est absente ou peu fiable.

    \item \textbf{Ajustement du normaliseur uniquement sur l'ensemble
    d'entraînement}, puis application (\texttt{transform}) à l'ensemble
    de test, afin d'éliminer toute fuite d'information statistique du test
    vers l'entraînement.

    \item \textbf{Validation croisée K-Fold aléatoire} (5 plis,
    \texttt{shuffle=True}) pour l'optimisation des hyperparamètres via
    \textit{RandomizedSearchCV}, les 200 échantillons correspondant à des
    points spatiaux indépendants et non à une série temporelle séquentielle.
\end{enumerate}

\section{Résultats après correction}

Le Tableau~\ref{tab:performance_spi_corrige} présente les performances
obtenues après suppression de la fuite de données.

\begin{table}[H]
\centering
\small
\caption{Performance des modèles en entraînement et test, après correction
de la fuite de données}
\label{tab:performance_spi_corrige}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Modèle} &
\textbf{MSE Train} &
\textbf{RMSE Train} &
\textbf{$R^2$ Train} &
\textbf{MSE Test} &
\textbf{RMSE Test} &
\textbf{$R^2$ Test} \\
\midrule
Random Forest & 0,04380 & 0,20927 & 0,9583 & 0,13866 & 0,37238 & 0,8205 \\
XGBoost       & 0,01429 & 0,11956 & 0,9864 & 0,18860 & 0,43429 & 0,7559 \\
SVR           & 0,18426 & 0,42925 & 0,8246 & 0,32597 & 0,57094 & 0,5781 \\
\bottomrule
\end{tabular}
\end{table}

\subsection*{Interprétation}

Après correction, le pouvoir prédictif réel des indices satellite sur le
SPI se situe dans une fourchette réaliste ($R^2$ test entre 0,58 et 0,82),
bien inférieure aux valeurs proches de 0,999 rapportées initialement, mais
scientifiquement significative.

Ces résultats montrent qu'une part substantielle de la variance du SPI
peut être expliquée par les seules données satellitaires, sans recours à
la mesure directe de précipitation. La hiérarchie entre modèles est
préservée : Random Forest demeure le plus performant sur le jeu de test
($R^2=0.8205$), suivi de XGBoost ($R^2=0.7559$), puis de SVR
($R^2=0.5781$).

L'écart entre les performances d'entraînement et de test est de 0,1378
pour Random Forest, 0,2305 pour XGBoost et 0,2465 pour SVR, ce qui indique
un surapprentissage plus marqué pour XGBoost et SVR que pour Random Forest.

\section{Code source corrigé}

Le script Python correspondant doit être placé dans le répertoire
\texttt{code/}, par exemple sous le nom
\texttt{spi\_satellite\_leakage\_free.py}.

Pour l'inclure dans le document final :

\begin{verbatim}
\lstinputlisting[style=pythonstyle,
caption={Pipeline SPI corrigé et sans fuite de données}]
{code/spi_satellite_leakage_free.py}
\end{verbatim}

\section{Traçabilité des corrections}

Les scripts Python corrigés doivent être versionnés avec ce document dans
le même dépôt GitHub. Il est recommandé de conserver les scripts originaux
dans un répertoire distinct, par exemple \texttt{code/archive/}, afin de
préserver la traçabilité des corrections méthodologiques.

\begin{itemize}
    \item pipeline SHI corrigé :
    \texttt{code/shi\_lucas2018\_leakage\_free.py} ;

    \item pipeline BiLSTM--CatBoost et test de Diebold--Mariano :
    \texttt{code/bilstm\_catboost\_dm\_leakage\_free.py} ;

    \item pipeline SPI corrigé :
    \texttt{code/spi\_satellite\_leakage\_free.py}.
\end{itemize}

\end{document}
"""

tex_path = base / "annexe_shi_spi_leakage_free.tex"
tex_path.write_text(tex, encoding="utf-8")

readme = """# Pipelines corrigés — SHI, BiLSTM–CatBoost et SPI

Ce dépôt accompagne l'annexe méthodologique de la thèse et contient les
pipelines Python corrigés pour éviter les fuites de données (`data leakage`).

## Fichiers

- `annexe_shi_spi_leakage_free.tex` : document LaTeX de l'annexe.
- `code/shi_lucas2018_leakage_free.py` : pipeline SHI corrigé.
- `code/bilstm_catboost_dm_leakage_free.py` : comparaison BiLSTM–CatBoost et test de Diebold–Mariano.
- `code/spi_satellite_leakage_free.py` : pipeline SPI corrigé.

## Principes méthodologiques

Les corrections portent notamment sur :

- séparation stricte entraînement/test ;
- ajustement des imputateurs et normaliseurs uniquement sur l'entraînement ;
- absence de réutilisation des composantes de la cible comme prédicteurs ;
- sélection des modèles/hyperparamètres sans utiliser le jeu de test ;
- validation temporelle `TimeSeriesSplit` pour les expériences temporelles ;
- exclusion des observations synthétiques des métriques finales lorsque le protocole l'exige ;
- audit automatique de fuite de données ;
- explicabilité SHAP réalisée après verrouillage du modèle final.

## Important

Les scripts Python doivent être ajoutés au répertoire `code/`. Ils ne sont pas
reconstitués automatiquement à partir de l'annexe : il faut déposer ici les
versions corrigées effectivement exécutées et utilisées pour produire les
résultats rapportés dans la thèse.

## Compilation

Compiler `annexe_shi_spi_leakage_free.tex` avec XeLaTeX, LuaLaTeX ou pdfLaTeX
selon le préambule du projet principal.
"""
(base / "README.md").write_text(readme, encoding="utf-8")

code_dir = base / "code"
code_dir.mkdir(exist_ok=True)
(code_dir / "README.txt").write_text(
    "Déposer ici les scripts Python corrigés effectivement exécutés.\n"
    "Noms recommandés :\n"
    "- shi_lucas2018_leakage_free.py\n"
    "- bilstm_catboost_dm_leakage_free.py\n"
    "- spi_satellite_leakage_free.py\n",
    encoding="utf-8"
)

# Zip the package for easy GitHub preparation
import shutil
zip_path = shutil.make_archive("/mnt/data/annexe_shi_spi_leakage_free", "zip", base)

print(tex_path)
print(zip_path)
