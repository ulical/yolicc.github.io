
<?php
$str_1 = crc32("m1");
echo 'Without %u: '.$str_1."<br>";
echo 'With %u: ';


$str_2 = crc32("m2");
echo 'Without %u_1: '.$str_2."<br>";
echo 'With %u_1: ';


  $x=$str_1;
  $y=$str_2;
  $d=$x-$y;
  
  if($d<0)
  {
	  $d=-$d;
  }
    if($d==0)
  {
	  print("亲密无间");
  }
    elseif($d==1)
  {
	  print("永远和你在一起");
  }
    elseif($d==2)
  {
	  print("水火不相容");
  }
    elseif($d==3)
  {
	  print("知心朋友");
  }
    elseif($d==4)
  {
	  print(" 心上人");
  } elseif($d==5)
  {
	  print("帮你做事的人");
   }elseif($d==6)
  {
	  print("帮你的人 ");
  } elseif($d==7)
  {
	  print("面和心不合");
  } elseif($d==8)
  {
	  print("男女关系不正常");
  } elseif($d==9)
  {
	  print("情投意合 ");
  } elseif($d==10)
  {
	  print("关系马虎");
  } elseif($d==11)
  {
	  print("尊敬你的人 ");
  } elseif($d==2)
  {
	  print("爱你的人");
  } elseif($d==13)
  {
	  print("适合你的");
  } elseif($d==14)
  {
	  print("说你坏话的人");
  } elseif($d==15)
  {
	  print("克星");
  } elseif($d==16)
  {
	  print("救星");
  } elseif($d==17)
  {
	  print("忠心的人");
  } elseif($d==18)
  {
	  print("狼心狗肺的人");
  } elseif($d==19)
  {
	  print("单相思");
  } elseif($d==20)
  {
	  print("山盟海誓 ");
  } elseif($d==21)
  {
	  print("情敌");
  } elseif($d==22)
  {
	  print("服从你的人");
  } elseif($d==23)
  {
	  print("永远在一起 ");
  } elseif($d==24)
  {
	  print("伴终生");
  } elseif($d==25)
  {
	  print("恨你又爱你");
  } else if ($d>25)
  {
	  print("你俩缘分超出计算范围");
  }


?>
