<?php   
$input=$_POST["logo"];
exec('python process.py $input');
header("Location: finaloutput.html");

?>