<?php   
$input=$_POST["logo"];
exec('python detectMultiLogo_cam.py $input');
header("Location: finaloutput.html");

?>