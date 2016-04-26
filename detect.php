<?php
//echo "<pre>$output</pre>";
//$input=$_POST["logo"];
//echo $input;
shell_exec('python svm_video.py');
header("Location: finalprocess.html");
//2>&1
//foreach($output as $line)
//{
//   echo "$line \n"; 
//}


?>