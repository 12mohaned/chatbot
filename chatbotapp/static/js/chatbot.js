var message = "";
// Responsible for sending message to the chatbot.
function Send(){
  var textAreaValue = document.getElementsByTagName("textarea")[0].value;
  if(textAreaValue.length > 2){
    appendtoLog(textAreaValue);
    document.getElementById("textArea").value = "";
  }
}

function setLastMessage(message){
  this.message = message;
}

function getLastMessage(){
  return message;
}

//Add message(s) to the chat Log
function appendtoLog(message){
  var user_friend = document.createElement('div');
  user_friend.className = "chat friend";
  var user_message = document.createElement("p");
  user_message.className = "user-message";
  var value = document.createTextNode(message);
  user_message.appendChild(value);
  user_friend.appendChild(user_message);
  document.getElementsByClassName("chatlogs")[0].appendChild(user_friend);
  setLastMessage(value);
}
