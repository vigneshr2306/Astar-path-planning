int lm1=5,lm2=3,rm1=6,rm2=9;

void setup() 
{
  // LM1, RM1 is Positive
  // LM2, RM2 is negative pin
pinMode(lm1,OUTPUT);
pinMode(lm2,OUTPUT);
pinMode(rm1,OUTPUT);
pinMode(rm2,OUTPUT);
Serial.begin(115200);
}

void loop() {
  // Recieving data from python.
  // If recieved data is
  // 0 --> Stop
  // 1 --> Forward
  // 2 --> Right
  // 3 --> Left
  // 4 --> Slow Right
  // 5 --> Slow Left
  // 6 --> Reverse
if(Serial.available()>0){
  int a;
  a=Serial.read();
 // Serial.println(a);
  if(a==0)
  {
    stop1();
  }
  else if(a==1)
  {
    forward();
  }
  else if(a==2)
  {
    right();
  }
  else if(a==3)
  {
    left();
  }
  else if(a==4)
  {
    iright();
  }
  else if(a==5)
  {
    ileft();
  }
  else if(a==6)
  {
    reverse();
  }
 }
// else 
// {
// stop1();
// }
}
void forward()
{
  analogWrite(lm1,150);
  analogWrite(lm2,0);
  analogWrite(rm1,150);
  analogWrite(rm2,0);
        
  }

  void left()
  {
  analogWrite(lm1,0);
  analogWrite(lm2,150);
  analogWrite(rm1,150);
  analogWrite(rm2,0);        
  }

  void right()
  {
  analogWrite(lm1,150);
  analogWrite(lm2,0);
  analogWrite(rm1,0);
  analogWrite(rm2,150);        
  }

  void back()
  {
  analogWrite(lm2,150);
  analogWrite(rm1,0);
  analogWrite(rm2,150);        
  }
  
  void stop1()
  {
  analogWrite(lm1,150);
  analogWrite(lm2,150);
  analogWrite(rm1,150);
  analogWrite(rm2,150);    
  }
  void iright()
 {
  analogWrite(lm1,150);
  analogWrite(lm2,0);
  analogWrite(rm1,0);
  analogWrite(rm2,0);
 }
void ileft()
{
  analogWrite(lm1,0);
  analogWrite(lm2,0);
  analogWrite(rm1,150);
  analogWrite(rm2,0);
        
  }
  void reverse()
  {  
  analogWrite(lm1,0);
  analogWrite(lm2,150);
  analogWrite(rm1,0);
  analogWrite(rm2,150);        
  }
 
