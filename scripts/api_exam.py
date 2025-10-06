import requests
import pandas as pd

# (a) À l'aide d'une requête GET sur l'endpoint "https://examen-api.s3.eu-west-1.amazonaws.com/Students", 
# récupérer la liste des identifiants des étudiants et stocker le résultat dans une liste nommée student_list.

# Requête GET pour récupérer les données du endpoint
response = requests.get("https://examen-api.s3.eu-west-1.amazonaws.com/Students")

# Affichage du code de statut
print(response.status_code)

# Chargement de la réponse
data = (response.json())                        
                        
# Création de la liste
student_list = data['StudentList']

# Verification du tableau
print(student_list)

# (b) Définir une fonction nommée extract_enrollments 
# qui va prendre en argument une liste d'identifiants d'apprenants 
# et renvoyer un dataframe CourseID 	StudentID 	StudentName 	StudentCursus
def extract_enrollments(student_list):
    # chaine url et dictionnaire de données
    url_chain="{url}/{endpoint}"
    # Gestion des données de retour
    data = []
    url = "https://examen-api.s3.eu-west-1.amazonaws.com/Student"
    for student_id in student_list:
        # Requête GET pour récupérer les données du endpoint
        response = requests.get(url_chain.format(url=url, endpoint=student_id))
        # Chargement de la réponse
        data.append(response.json())
    
    # ordre des colonnes du dataframe
    order = ['CourseID', 'StudentID', 'StudentName', 'StudentCursus']
    # dictionnaire pour renommage
    rename_dict = {
        'StudentCourses': 'CourseID'
    }
    # Conversion en DataFrame
    df = pd.DataFrame(data)
    # On met à plat la liste des cours
    df = df.explode("StudentCourses")
    # Renommage des colonnes
    df = df.rename(columns=rename_dict)
    # Réorganisation de l'ordre des colonnes
    df = df.reindex(columns=[rename_dict.get(col, col) for col in order])
    # retour du dataframe
    return df


# Chargement des données dans un dataframe
enrollments = extract_enrollments(student_list)
# verification du dataframe
print(enrollments.info())
# Changement du type en int (object intialement)
enrollments['CourseID'] = enrollments['CourseID'].astype('int64')
# verification du dataframe
print(enrollments.info())
print(enrollments.head())


# (d) Définir une fonction nommée extract_attendances qui va prendre en argument une liste d'identifiants d'apprenants
#  et renvoyer le DataFrame suivant: # CourseID 	StudentID 	Date
def extract_attendances(student_list):
    # chaine url et dictionnaire de données
    url_chain="{url}/{endpoint}"
    # Gestion des données de retour
    data = []
    url = "https://examen-api.s3.eu-west-1.amazonaws.com/Attendance"
    for student_id in student_list:
        # Requête GET pour récupérer les données du endpoint
        response = requests.get(url_chain.format(url=url, endpoint=student_id))
        # Chargement de la réponse
        data.append(response.json())

    # tableau pour le dataframe
    data_for_df = []
    for dict_student in data:
        for dict_studentAttendance in dict_student['StudentAttendance']:
            data_for_df.append(
                { 
                    'CourseID'  : list(dict_studentAttendance.keys())[0],
                    'StudentID' : dict_student['StudentID'], 
                    'Date'      : list(dict_studentAttendance.values())[0] 
                }
            )
    # Conversion en DataFrame
    df = pd.DataFrame(data_for_df)
    # retour du dataframe
    return df


# Chargement des données dans un dataframe
attendances = extract_attendances(student_list)
# verification du dataframe
print(attendances.info())
# Changement du type en int (object intialement)
attendances['CourseID'] = attendances['CourseID'].astype('int64')
print(attendances.info())
print(attendances.head(10))


# (f) Définir une fonction nommée extract_grades qui va prendre en argument une liste d'identifiants d'apprenants
#  et renvoyer le DataFrame suivant: # CourseID 	StudentID 	Grade 	Attended 	Success
def extract_grades(student_list):
    # chaine url et dictionnaire de données
    url_chain="{url}/{endpoint}"
    # Gestion des données de retour
    data = []
    url = "https://examen-api.s3.eu-west-1.amazonaws.com/Grades"
    for student_id in student_list:
        # Requête GET pour récupérer les données du endpoint
        response = requests.get(url_chain.format(url=url, endpoint=student_id))
        # Chargement de la réponse
        data.append(response.json())

    # tableau pour le dataframe
    data_for_df = []
    for dict_student in data:
        for dict_studentGrades in dict_student['StudentGrades']:
            data_for_df.append(
                { 
                    'CourseID'  : list(dict_studentGrades.keys())[0],
                    'StudentID' : dict_student['StudentID'], 
                    'Grade'     : list(dict_studentGrades.values())[0],
                    'Attended'  : True,
                    'Success'   : True
                    
                }
            )
    # Conversion en DataFrame
    df = pd.DataFrame(data_for_df)
    # Attended / Sucess
    df['Attended'] = df['Grade'] > 0
    df['Success'] = df['Grade'] >= 10
    # retour du dataframe
    return df

# (g) Exécuter la fonction extract_grades avec en argument la liste student_list
#  et stocker le résultat produit dans un DataFrame nommé grades.
grades = extract_grades(student_list)
print(grades.info())
# Changement du type en int (object intialement)
grades['CourseID'] = grades['CourseID'].astype('int64')
print(grades.info())
print(grades.head(10))

# II. Calcul de KPIs

# (a) Définir une fonction nommée transform_enrollments qui va prendre en argument le DataFrame enrollments et calculer les KPIs suivants :
# Nombre d'apprenants inscrits dans une colonne nommée "EnrolledStudents".
# Cursus le plus fréquent parmi les apprenants dans une colonne nommée "MajorityCursus".

# Indication : Vous pouvez renommer les colonnes d'un DataFrame en utilisant la méthode suivante:
# df = df.rename(columns = {
#     "Nom de la colonne" : "Nouveau nom",
#     ...
#     })
def transform_enrollments(enrollments):
    # Nombre d'apprenants inscrits par cours
    enrolledStudents = enrollments.groupby('CourseID').agg(EnrolledStudents=('StudentID', 'count'))
    # Cursus majoritaire par cours (mode)
    majorityCursus = enrollments.groupby('CourseID')['StudentCursus'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    # Fusion des deux KPIs dans un seul DataFrame
    result = enrolledStudents.copy()
    result['MajorityCursus'] = majorityCursus
    return result.reset_index()


# (b) Tester la fonction transform_enrollments. On devrait obtenir le DataFrame suivant :
# CourseID 	EnrolledStudents 	MajorityCursus
df_enrollments = transform_enrollments(enrollments)

print(df_enrollments.head())

# (c) Définir une fonction nommée transform_attendances qui va prendre en argument le DataFrame attendances 
# et calculer le taux de présence aux séances de chaque cours. On stockera le résultat dans une colonne nommée "AttendanceRate".
# Indications :
# Chaque cours est décomposé en 10 séances. 
# Pour calculer le taux de présence à un cours, il suffit de compter le nombre de séances où chaque apprenant a été présent par cours, puis le diviser par 10.
# Dans la méthode groupby, il est possible de grouper les individus selon plusieurs variables. 
# Par exemple attendances.groupby(["StudentID", "CourseID"]) pour grouper par apprenant puis par cursus.
# Pensez à utiliser la méthode reset_index pour retirer de l'index les colonnes utilisées pour une opération groupby.
def transform_attendances(attendances):
    # 1. Présences par étudiant (count par CourseID/StudentID)
    presences_indiv = attendances.groupby(['CourseID', 'StudentID']).size().reset_index(name='PresCount')
    # 2. Somme des présences (toutes les présences de tous les étudiants pour chaque CourseID)
    total_presences = presences_indiv.groupby('CourseID')['PresCount'].sum()
    # 3. Nombre d'étudiants uniques par cours
    num_students = presences_indiv.groupby('CourseID')['StudentID'].nunique()
    # 4. Taux de présence par cours
    taux_presence = total_presences / (num_students * 10)
    # 5. Conversion en dataframe
    result = taux_presence.reset_index(name='AttendanceRate')
    return result

# (d) Tester la fonction transform_attendances, on devrait obtenir le DataFrame suivant :
# CourseID 	AttendanceRate
df_attendances = transform_attendances(attendances)
print(df_attendances.head())

# (e) Définir une fonction nommée transform_grades qui va prendre en argument le DataFrame grades et calculer les KPIs suivants:
# CourseID 	StudentID 	Grade 	Attended 	Success
# Taux de participation à l'examen dans une colonne nommée "ExamAttendanceRate". On supposera qu'un apprenant a été présent à l'examen si sa note est strictement supérieure à 0.
# Taux de réussite à l'examen dans une colonne nommée "ExamSuccessRate". On supposera qu'un apprenant a réussi l'examen si sa note est supérieure ou égale à 10.
# Moyenne des notes dans une colonne nommée "ExamAverage". On ne comptera dans cette moyenne que les notes strictement supérieures à 0.
# On commencera par définir dans grades une colonne indiquant si l'apprenant a été présent à l'examen et une colonne indiquant s'il a réussi l'examen.
# Pour calculer la moyenne à l'examen, on pourra pour chaque cours d'abord calculer la somme des notes puis diviser cette somme par le nombre d'étudiants présents à l'examen.
def transform_grades(grades):
    # KPIs par CourseID
    # Avec les colonnes existantes Attended et Success 
    # Définies précédemment avec les règles de note > 0 et >= 10
    result = grades.groupby('CourseID').agg(
        ExamAttendanceRate=('Attended', 'mean'),
        ExamSuccessRate=('Success', 'mean'),
        ExamAverage=('Grade', lambda x: x[x > 0].mean())
    ).reset_index()

    return result

# (f) Tester la fonction transform_grades. On devrait obtenir le DataFrame suivant:
# CourseID 	ExamAttendanceRate 	ExamSuccessRate 	ExamAverage
df_grades = transform_grades(grades)
print(df_grades.head())

# (g) À l'aide des fonctions définies précédemment, effectuer une fusion permettant d'obtenir le DataFrame suivant :
# CourseID 	EnrolledStudents 	MajorityCursus 	AttendanceRate 	ExamAttendanceRate 	ExamSuccessRate 	ExamAverage
df_merged = df_enrollments.merge(df_attendances, on='CourseID', how='left').merge(df_grades, on='CourseID', how='left')  
print(df_merged.head())
# (h) À votre avis, quels sont les examens trop difficiles? Quels sont les examens trop faciles?
# sur le DataFrame df_merged, on peut regarder les taux de réussite (ExamSuccessRate) et les moyennes (ExamAverage)
# en classant les cours par ordre croissant de taux de réussite et de moyenne dans un tableau   
print("Classement des 3 examens les plus difficiles en premiers\n",
      df_merged[['CourseID', 'ExamSuccessRate', 'ExamAverage']]
        .sort_values(by=['ExamSuccessRate', 'ExamAverage'], ascending=[True, True]).head(3))

print("Classement des 3 examens les plus faciles en premiers\n",
      df_merged[['CourseID', 'ExamSuccessRate', 'ExamAverage']]
        .sort_values(by=['ExamSuccessRate', 'ExamAverage'], ascending=[False, False]).head(3))

