// Responsible for sending message to the chatbot.
function Send(){
  var textAreaValue = document.getElementsByTagName("textarea")[0].value;
  if(textAreaValue.length > 2){
    appendtoLog(textAreaValue);
    document.getElementById("textArea").value = "";
  }
}

function appendtoLog(message){
  user_friend = document.createElement('div');
  user_friend.className = "chat friend";
  user_message = document.createElement("p");
  user_message.className = "user-message";
  var value = document.createTextNode(message);
  user_message.appendChild(value);
  user_friend.appendChild(user_message);
  document.getElementsByClassName("chatlogs")[0].appendChild(user_friend);
}
