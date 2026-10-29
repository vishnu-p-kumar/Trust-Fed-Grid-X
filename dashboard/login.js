(function () {
  const apiBase = window.location.origin.replace(/\/$/, "");

  async function checkExistingSession() {
    const response = await fetch(`${apiBase}/auth/session`, { credentials: "include" });
    if (!response.ok) {
      return null;
    }
    return response.json();
  }

  function showError(message) {
    const errorBox = document.getElementById("loginError");
    errorBox.textContent = message;
    errorBox.classList.add("visible");
  }

  function setSubmittingState(isSubmitting) {
    const submitButton = document.querySelector('#loginForm button[type="submit"]');
    if (!submitButton) {
      return;
    }
    submitButton.disabled = isSubmitting;
    submitButton.textContent = isSubmitting ? "Signing In..." : "Login";
  }

  async function init() {
    try {
      const session = await checkExistingSession();
      if (session?.authenticated) {
        window.location.href = session.role === "admin" ? "/admin" : "/inspector";
        return;
      }
    } catch (error) {
      return;
    }

    document.getElementById("loginForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const formData = new FormData(form);
      const identifier = String(formData.get("identifier") || "").trim();
      const errorBox = document.getElementById("loginError");
      errorBox.classList.remove("visible");
      setSubmittingState(true);
      try {
        const response = await fetch(`${apiBase}/auth/login`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            identifier,
            password: formData.get("password"),
          }),
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({ detail: "Invalid login." }));
          showError(payload.detail || "Invalid login.");
          setSubmittingState(false);
          return;
        }
        const fallbackRedirect = identifier.includes("@") ? "/admin" : "/inspector";
        window.location.replace(fallbackRedirect);
      } catch (error) {
        showError("Unable to sign in right now.");
        setSubmittingState(false);
      }
    });
  }

  init();
})();
