const resizer = document.getElementById("resizer");
const left = document.getElementById("leftPanel");

let isDragging = false;

resizer.addEventListener("mousedown", () => {
  isDragging = true;
  document.body.style.cursor = "col-resize";
});

window.addEventListener("mousemove", (e) => {
  if (!isDragging) return;

  let width = e.clientX;

  if (width < 200) width = 200;
  if (width > 500) width = 500;

  left.style.width = width + "px";
});

window.addEventListener("mouseup", () => {
  isDragging = false;
  document.body.style.cursor = "default";
});
