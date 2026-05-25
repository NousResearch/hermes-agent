/*
INSTRUCTIONS:
Handles sidebar toggle behavior.
*/

const sidebar = document.getElementById("sidebar");
const toggle = document.getElementById("toggleSidebar");

toggle.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
});
