const input = document.getElementById("chatInput");
const btn = document.getElementById("sendBtn");
const chatWindow = document.getElementById("chatWindow");

function addMessage(text, isUser = false) {
  const div = document.createElement("div");
  div.className = "chat-message" + (isUser ? " chat-user" : "");
  div.textContent = text;

  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage() {
  const msg = input.value.trim();
  if (!msg) return;

  addMessage(msg, true);
  input.value = "";

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message: msg })
    });

    const data = await res.json();
    addMessage(data.response);

  } catch (err) {
    addMessage("Error: Unable to reach server");
  }
}

btn.addEventListener("click", sendMessage);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
