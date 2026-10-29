(function () {
  const sameOriginApi = window.location.origin.replace(/\/$/, "");
  let cachedAreas = null;
  let cachedInspectors = null;
  const fallbackAreas = [
    "Whitefield",
    "Electronic City",
    "Indiranagar",
    "Koramangala",
    "Marathahalli",
    "Yelahanka",
    "BTM Layout",
    "Jayanagar",
    "Malleshwaram",
    "Rajajinagar",
    "Hebbal",
    "Bellandur",
    "HSR Layout",
    "Banashankari",
    "Peenya Industrial Area",
  ];

  function showToast(message, level) {
    const container = document.createElement("div");
    container.className = `alert alert-${level || "info"} position-fixed top-0 end-0 m-3`;
    container.style.zIndex = "1080";
    container.textContent = message;
    document.body.appendChild(container);
    setTimeout(() => container.remove(), 2600);
  }

  async function fetchSession() {
    const response = await fetch(`${sameOriginApi}/auth/session`, { credentials: "include" });
    if (!response.ok) {
      throw new Error("Session unavailable");
    }
    return response.json();
  }

  async function fetchAreas() {
    if (cachedAreas) {
      return cachedAreas;
    }
    const response = await fetch(`${sameOriginApi}/api/inspection-areas`, { credentials: "include" });
    if (!response.ok) {
      throw new Error("Areas unavailable");
    }
    const payload = await response.json();
    cachedAreas = payload.areas || [];
    return cachedAreas;
  }

  async function fetchInspectors() {
    if (cachedInspectors) {
      return cachedInspectors;
    }
    const response = await fetch(`${sameOriginApi}/api/inspectors`, { credentials: "include" });
    if (!response.ok) {
      throw new Error("Inspectors unavailable");
    }
    const payload = await response.json();
    cachedInspectors = payload.inspectors || [];
    return cachedInspectors;
  }

  function normalizeAreaOptions(areas) {
    return (areas || [])
      .map((area) => {
        if (typeof area === "string") {
          return area.trim();
        }
        if (area && typeof area === "object") {
          return String(area.area || area.name || area.label || area.value || "").trim();
        }
        return "";
      })
      .filter(Boolean);
  }

  function formatApiErrorDetail(detail) {
    if (!detail) {
      return "Unable to create inspector.";
    }
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (!item || typeof item !== "object") {
            return String(item || "").trim();
          }
          const path = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(" -> ") : "";
          const message = String(item.msg || item.message || "").trim();
          return [path, message].filter(Boolean).join(": ");
        })
        .filter(Boolean);
      return messages.join("\n") || "Unable to create inspector.";
    }
    if (typeof detail === "object") {
      return String(detail.message || detail.error || JSON.stringify(detail));
    }
    return String(detail);
  }

  function formatNotificationMessage(notification, inspector) {
    if (!notification || typeof notification !== "object") {
      return "Inspector created successfully.";
    }
    const label = inspector && (inspector.name || inspector.username) ? (inspector.name || inspector.username) : "Inspector";
    if (notification.status === "sent") {
      return `Inspector ${label} created and Telegram message sent.`;
    }
    if (notification.status === "skipped") {
      return `Inspector ${label} created, but Telegram message was skipped. ${notification.detail || ""}`.trim();
    }
    if (notification.status === "error") {
      return `Inspector ${label} created, but Telegram message failed. ${notification.detail || ""}`.trim();
    }
    return `Inspector ${label} created successfully.`;
  }

  function inspectorRowsMarkup(inspectors) {
    if (!inspectors.length) {
      return '<div class="empty-state">No inspectors created yet.</div>';
    }
    return `
      <div class="table-responsive">
        <table class="section-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Username</th>
              <th>Assigned Area</th>
              <th>Telegram Chat ID</th>
              <th>Role</th>
              <th>Created</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${inspectors
              .map(
                (item) => `
                  <tr>
                    <td>${item.name || "-"}</td>
                    <td>${item.username || "-"}</td>
                    <td>${item.assigned_area || "-"}</td>
                    <td>${item.chat_id || "-"}</td>
                    <td>${item.role || "inspector"}</td>
                    <td>${(item.created_at || "").replace("T", " ").slice(0, 19) || "-"}</td>
                    <td><button class="table-action-button delete-inspector" data-username="${item.username}">Delete</button></td>
                  </tr>
                `,
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function renderInspectorRows(inspectors) {
    return inspectorRowsMarkup(inspectors);
  }

  async function openInspectorsModal() {
    const wrapper = document.createElement("div");
    wrapper.className = "modal fade";
    wrapper.tabIndex = -1;
    wrapper.innerHTML = `
      <div class="modal-dialog modal-xl modal-dialog-centered">
        <div class="modal-content glass-panel text-light">
          <div class="modal-header border-secondary-subtle">
            <div>
              <p class="eyebrow mb-1">Inspector Directory</p>
              <h5 class="modal-title mb-0">Registered Inspectors</h5>
            </div>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body" id="inspectorsModalBody"><div class="empty-state">Loading inspectors...</div></div>
        </div>
      </div>
    `;
    document.body.appendChild(wrapper);
    const modal = new bootstrap.Modal(wrapper);
    wrapper.addEventListener("hidden.bs.modal", () => wrapper.remove());
    modal.show();
    try {
      const inspectors = await fetchInspectors();
      const body = wrapper.querySelector("#inspectorsModalBody");
      if (body) {
        body.innerHTML = renderInspectorRows(inspectors);
      }
    } catch (error) {
      const body = wrapper.querySelector("#inspectorsModalBody");
      if (body) {
        body.innerHTML = '<div class="empty-state">Unable to load inspectors right now.</div>';
      }
      showToast("Unable to load inspectors.", "danger");
    }
    wrapper.addEventListener("click", async (event) => {
      const button = event.target.closest(".delete-inspector");
      if (!button) {
        return;
      }
      const username = button.dataset.username;
      const deleteResponse = await fetch(`${sameOriginApi}/api/inspectors/${encodeURIComponent(username)}`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!deleteResponse.ok) {
        showToast("Unable to delete inspector.", "danger");
        return;
      }
      cachedInspectors = (cachedInspectors || []).filter((item) => item.username !== username);
      button.closest("tr").remove();
      showToast(`Inspector ${username} deleted.`, "success");
    });
  }

  async function openCreateModal() {
    const wrapper = document.createElement("div");
    wrapper.className = "modal fade";
    wrapper.tabIndex = -1;
    wrapper.innerHTML = `
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content glass-panel text-light">
          <form id="createInspectorForm">
            <div class="modal-header border-secondary-subtle">
              <div>
                <p class="eyebrow mb-1">Admin Action</p>
                <h5 class="modal-title mb-0">Create Inspector</h5>
              </div>
              <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
              <div class="mb-3">
                <label class="form-label">Name</label>
                <input class="form-control" name="name" required />
              </div>
              <div class="mb-3">
                <label class="form-label">Username</label>
                <input class="form-control" name="username" required />
              </div>
              <div class="mb-3">
                <label class="form-label">Assigned Area</label>
                <select class="form-select" name="assigned_area" required id="assignedAreaSelect">
                  <option value="">Loading areas...</option>
                </select>
              </div>
              <div class="mb-0">
                <label class="form-label">Password</label>
                <input class="form-control" name="password" type="password" required />
              </div>
              <div class="mt-3">
                <label class="form-label">Telegram Chat ID</label>
                <input class="form-control" name="chat_id" required inputmode="numeric" placeholder="1242950500" />
              </div>
              <div class="alert alert-danger mt-3 d-none" id="createInspectorError"></div>
            </div>
            <div class="modal-footer border-secondary-subtle">
              <button type="button" class="btn btn-outline-light" data-bs-dismiss="modal">Cancel</button>
              <button type="submit" class="btn btn-info">Create</button>
            </div>
          </form>
        </div>
      </div>
    `;
    document.body.appendChild(wrapper);
    const modal = new bootstrap.Modal(wrapper);
    wrapper.addEventListener("hidden.bs.modal", () => wrapper.remove());
    modal.show();
    const areaSelect = wrapper.querySelector("#assignedAreaSelect");
    const setAreaOptions = (areas) => {
      const normalizedAreas = normalizeAreaOptions(areas);
      areaSelect.innerHTML = `<option value="">Select area</option>${normalizedAreas.map((area) => `<option value="${area}">${area}</option>`).join("")}`;
    };
    setAreaOptions(cachedAreas && cachedAreas.length ? cachedAreas : fallbackAreas);
    try {
      const areas = await fetchAreas();
      setAreaOptions(areas.length ? areas : fallbackAreas);
    } catch (error) {
      showToast("Using fallback area list for inspector creation.", "warning");
    }
    wrapper.querySelector("#createInspectorForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const errorBox = wrapper.querySelector("#createInspectorError");
      errorBox.classList.add("d-none");
      const formData = new FormData(form);
      const response = await fetch(`${sameOriginApi}/api/inspectors`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formData.get("name"),
          username: formData.get("username"),
          assigned_area: formData.get("assigned_area"),
          password: formData.get("password"),
          chat_id: formData.get("chat_id"),
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ detail: "Unable to create inspector." }));
        errorBox.textContent = formatApiErrorDetail(payload.detail);
        errorBox.classList.remove("d-none");
        return;
      }
      const payload = await response.json();
      cachedInspectors = [...(cachedInspectors || []), payload.inspector].sort((a, b) => `${a.name || ""}${a.username || ""}`.localeCompare(`${b.name || ""}${b.username || ""}`));
      modal.hide();
      showToast(
        formatNotificationMessage(payload.notification, payload.inspector),
        payload.notification && payload.notification.status !== "sent" ? "warning" : "success"
      );
    });
  }

  async function logout() {
    window.location.href = "/login";
    fetch(`${sameOriginApi}/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
  }

  async function init() {
    if (!window.location.pathname.endsWith("/dashboard/index.html")) {
      return;
    }
    try {
      const session = await fetchSession();
      if (session.role !== "admin") {
        window.location.href = session.role === "inspector" ? "/inspector" : "/login";
        return;
      }
      const adminActions = document.getElementById("adminActions");
      if (adminActions) {
        adminActions.classList.remove("d-none");
      }
      fetchAreas().catch(() => null);
      fetchInspectors().catch(() => null);
      document.getElementById("createInspectorButton")?.addEventListener("click", openCreateModal);
      document.getElementById("viewInspectorsButton")?.addEventListener("click", openInspectorsModal);
      document.getElementById("logoutButton")?.addEventListener("click", logout);
    } catch (error) {
      window.location.href = "/login";
    }
  }

  init();
})();
