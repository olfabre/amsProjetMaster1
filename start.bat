@echo off
:: Définir le nom du job
set JOB_NAME=TP3

:: Configuration des ressources (non applicable directement sous Windows)
:: Windows ne supporte pas directement SLURM, donc ces options sont ignorées.
:: Pour des tâches GPU, il faut s'assurer que les pilotes et CUDA sont configurés.

:: Création des dossiers de logs si nécessaire
if not exist logs mkdir logs

:: Rediriger la sortie et les erreurs
set LOG_OUT=logs\%JOB_NAME%_out.log
set LOG_ERR=logs\%JOB_NAME%_err.log

:: Lancer le script Python et rediriger les sorties
echo Lancement de artGen.py...
python artGen.py > %LOG_OUT% 2> %LOG_ERR%

:: Afficher un message une fois terminé
echo Script terminé. Logs enregistrés dans %LOG_OUT% et %LOG_ERR%.
pause
