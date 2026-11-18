"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const MIN = Math.max(5, Number($("main").dataset.minRatings) || 5);
  const genres = {
    Action: "боевик",
    Adventure: "приключения",
    Animation: "анимация",
    "Children's": "семейный",
    Comedy: "комедия",
    Crime: "криминал",
    Documentary: "документальный",
    Drama: "драма",
    Fantasy: "фэнтези",
    "Film-Noir": "нуар",
    Horror: "ужасы",
    Musical: "мюзикл",
    Mystery: "детектив",
    Romance: "мелодрама",
    "Sci-Fi": "фантастика",
    Thriller: "триллер",
    War: "военный",
    Western: "вестерн",
  };
  const ratings = new Map();
  const skipped = new Set();
  const seen = new Set();
  const unfamiliar = new Set();
  let pool = [],
    queue = [],
    watchlist = [],
    searchItems = [];
  let ready = false,
    busy = false,
    view = "selection",
    choosing = false;
  let recSeq = 0,
    historySeq = 0,
    watchSeq = 0,
    searchSeq = 0,
    searchTimer;
  let recAbort,
    searchAbort,
    failedAction = null;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function button(text, action, className = "secondary") {
    const node = el("button", className, text);
    node.type = "button";
    node.addEventListener("click", action);
    return node;
  }
  function message(target, text, retry) {
    target.replaceChildren(el("p", "empty", text));
    if (retry) target.append(button("Попробовать снова", retry));
  }
  function announce(text) {
    $("status").textContent = text;
  }
  function errorCopy(err) {
    if (err.status === 429)
      return "Слишком много действий подряд. Подождите минуту и повторите.";
    if (err.status === 403)
      return "Достигнут лимит оценок. Сохранённые фильмы останутся с вами.";
    if (err.status === 409)
      return "Этот фильм уже оценён. Обновите оценки и повторите действие.";
    return "Не удалось связаться с сервисом. Проверьте соединение и попробуйте снова.";
  }
  async function request(url, options = {}) {
    const controller = new AbortController();
    const cancel = () => controller.abort();
    options.signal?.addEventListener("abort", cancel, { once: true });
    if (options.signal?.aborted) controller.abort();
    const timer = setTimeout(cancel, 15000);
    try {
      const response = await fetch(url, {
        ...options,
        credentials: "same-origin",
        signal: controller.signal,
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
        },
      });
      if (!response.ok) {
        const err = new Error("request");
        err.status = response.status;
        throw err;
      }
      return response.status === 204 ? null : await response.json();
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", cancel);
    }
  }
  function progress() {
    const n = ratings.size;
    $("progress-text").textContent =
      `${n} из ${MIN}${n >= MIN ? " · можно начинать" : " · ещё немного до вашей подборки"}`;
    $("progress-fill").style.width = `${Math.min(n / MIN, 1) * 100}%`;
    $("start-selection").disabled = !ready || busy || n < MIN;
  }
  function poster(movie, eager = false) {
    const frame = el("div", "poster");
    const fallback = () => {
      const placeholder = el("div", "poster-placeholder");
      placeholder.append(
        el("span", "poster-symbol", "▶"),
        el("span", "poster-name", movie.title),
        el("small", "", movie.year ?? ""),
      );
      frame.replaceChildren(placeholder);
    };
    // Only the trusted image endpoint is allowed, even with malformed server data.
    if (
      !/^https:\/\/image\.tmdb\.org\/t\/p\/w500\/[A-Za-z0-9_-]+\.(jpg|jpeg|png|webp)$/.test(
        movie.poster_url || "",
      )
    ) {
      fallback();
      return frame;
    }
    const image = el("img");
    image.alt = `Постер: ${movie.title}`;
    image.loading = eager ? "eager" : "lazy";
    image.decoding = "async";
    image.width = 500;
    image.height = 750;
    image.addEventListener("error", fallback, { once: true });
    image.src = movie.poster_url;
    frame.append(image);
    return frame;
  }
  function stars(movie) {
    const group = el("div", "stars");
    group.setAttribute("role", "radiogroup");
    group.setAttribute("aria-label", `Оценка фильма «${movie.title}»`);
    const current = ratings.get(movie.movie_id)?.user_rating || 0;
    for (let value = 1; value <= 5; value++) {
      const star = button(
        "★",
        () => rate(movie, value),
        `star${value <= current ? " on" : ""}`,
      );
      star.setAttribute("role", "radio");
      star.setAttribute("aria-checked", String(value === current));
      star.setAttribute("aria-label", `${value} из 5`);
      star.dataset.rating = value;
      star.tabIndex = value === (current || 1) ? 0 : -1;
      star.disabled = busy;
      star.addEventListener("keydown", (event) => {
        const delta = {
          ArrowRight: 1,
          ArrowUp: 1,
          ArrowLeft: -1,
          ArrowDown: -1,
        }[event.key];
        if (!delta && event.key !== "Home" && event.key !== "End") return;
        event.preventDefault();
        event.stopPropagation();
        const next =
          event.key === "Home"
            ? 1
            : event.key === "End"
              ? 5
              : Math.max(1, Math.min(5, value + delta));
        group.children[next - 1].focus();
        rate(movie, next);
      });
      group.append(star);
    }
    return group;
  }
  function movieCard(movie, kind) {
    const card = el("article", "movie-card");
    card.dataset.movieId = movie.movie_id;
    card.append(poster(movie));
    const body = el("div", "movie-body");
    body.append(
      el("h3", "movie-title", movie.title),
      el(
        "p",
        "movie-meta",
        [
          movie.year,
          ...(movie.genres || []).slice(0, 2).map((g) => genres[g] || g),
        ]
          .filter(Boolean)
          .join(" · "),
      ),
    );
    if (kind === "watchlist") {
      body.append(
        button(
          "Уже смотрел",
          () => {
            body.replaceChildren(
              ...Array.from(body.children).slice(0, 2),
              stars(movie),
            );
          },
          "text-button",
        ),
        button(
          "Убрать из списка",
          () => mutate(movie, "remove"),
          "text-button muted",
        ),
      );
    } else {
      body.append(stars(movie));
      if (kind === "onboarding")
        body.append(
          button(
            "Не смотрел",
            () => {
              if (busy) return;
              unfamiliar.add(movie.movie_id);
              renderOnboarding();
            },
            "text-button muted",
          ),
        );
    }
    card.append(body);
    return card;
  }
  function grid(id, movies, kind, empty) {
    if (!movies.length) {
      message($(id), empty);
      return;
    }
    $(id).replaceChildren(...movies.map((movie) => movieCard(movie, kind)));
  }
  function renderOnboarding() {
    grid(
      "onboarding-grid",
      pool.filter((m) => !unfamiliar.has(m.movie_id)).slice(0, 10),
      "onboarding",
      "Знакомые фильмы закончились. Найдите просмотренное по названию ниже.",
    );
  }
  function renderLists() {
    grid(
      "history-grid",
      [...ratings.values()],
      "history",
      "Здесь появятся ваши оценки. Найдите знакомый фильм ниже.",
    );
    grid(
      "watchlist-grid",
      watchlist,
      "watchlist",
      "Пока здесь пусто. Сохраните понравившийся фильм из подборки — он будет ждать вас здесь.",
    );
    renderOnboarding();
    grid(
      "search-grid",
      searchItems,
      "search",
      "Ничего не нашлось. Попробуйте другое название.",
    );
    if (!$("search").value.trim()) $("search-grid").replaceChildren();
    progress();
  }
  function lock() {
    document
      .querySelectorAll(".movie-card button, #deck button")
      .forEach((node) => {
        node.disabled = busy;
      });
    $("deck").setAttribute("aria-busy", String(busy));
    if (busy) {
      const pending = document.querySelector("#deck .card-error");
      if (pending) pending.replaceChildren(el("p", "", "Сохраняем ваш выбор…"));
    }
    progress();
  }
  async function loadHistory() {
    const mine = ++historySeq;
    let data;
    try {
      data = await request("/ratings");
    } catch (err) {
      if (mine !== historySeq) return false;
      throw err;
    }
    if (mine !== historySeq) return false;
    ratings.clear();
    for (const movie of data.items)
      if (movie.user_rating) ratings.set(movie.movie_id, movie);
    return true;
  }
  async function loadWatchlist() {
    const mine = ++watchSeq;
    let data;
    try {
      data = await request("/watchlist");
    } catch (err) {
      if (mine !== watchSeq) return;
      throw err;
    }
    if (mine !== watchSeq) return;
    watchlist = data.items;
    grid(
      "watchlist-grid",
      watchlist,
      "watchlist",
      "Пока здесь пусто. Сохраните фильм из подборки.",
    );
    lock();
  }
  async function loadOnboarding() {
    message($("onboarding-grid"), "Выбираем знакомые истории…");
    try {
      pool = (await request("/movies/onboarding")).items;
      renderOnboarding();
      lock();
    } catch (err) {
      message(
        $("onboarding-grid"),
        "Не удалось открыть фильмы.",
        loadOnboarding,
      );
    }
  }
  function allowed(movie) {
    return (
      !skipped.has(movie.movie_id) &&
      !seen.has(movie.movie_id) &&
      !ratings.has(movie.movie_id)
    );
  }
  async function loadRecommendations() {
    const mine = ++recSeq;
    recAbort?.abort();
    recAbort = new AbortController();
    if (!queue.length) message($("deck"), "Ищем историю для вашего вечера…");
    try {
      const data = await request("/recommendations?k=50", {
        signal: recAbort.signal,
      });
      if (mine !== recSeq) return;
      if (data.n_ratings === null || data.n_ratings === undefined)
        throw new Error("unavailable");
      const ids = new Set();
      queue = data.items.filter(
        (movie) =>
          allowed(movie) && !ids.has(movie.movie_id) && ids.add(movie.movie_id),
      );
      renderDeck();
    } catch (err) {
      if (mine !== recSeq) return;
      if (queue.length) {
        announce("Подборка временно не обновилась. Попробуйте позже.");
        renderDeck();
      } else
        message(
          $("deck"),
          "Подборка временно недоступна. Ваши оценки сохранены.",
          loadRecommendations,
        );
    }
  }
  function renderDeck() {
    queue = queue.filter(allowed);
    const movie = queue[0];
    if (!movie) {
      const empty = el("div", "deck-empty");
      empty.append(
        el("span", "empty-icon", "✦"),
        el("h2", "", "На сегодня всё"),
        el(
          "p",
          "subtle",
          "Вы просмотрели эту подборку. Найдите фильм ниже или вспомните ещё несколько любимых историй.",
        ),
        button("Вернуться к оценкам", () => navigate("history")),
      );
      $("deck").replaceChildren(empty);
      return;
    }
    const card = el("article", "deck-card");
    card.dataset.movieId = movie.movie_id;
    card.append(poster(movie, true));
    const body = el("div", "deck-body");
    body.append(
      el("p", "eyebrow", "Ваша следующая история"),
      el("h2", "deck-title", movie.title),
      el(
        "p",
        "movie-meta",
        [movie.year, ...(movie.genres || []).map((g) => genres[g] || g)]
          .filter(Boolean)
          .join(" · "),
      ),
    );
    const actions = el("div", "deck-actions");
    actions.append(
      button("× Не интересно", () => mutate(movie, "dismissed")),
      button("→ Пропустить", skip),
      button("♡ Хочу посмотреть", () => mutate(movie, "watchlist"), "primary"),
    );
    const watched = button(
      "Уже смотрел — оценить",
      () => {
        if (!busy) {
          ratingBox.hidden = !ratingBox.hidden;
          if (!ratingBox.hidden) ratingBox.querySelector("button").focus();
        }
      },
      "text-button",
    );
    const ratingBox = el("div", "deck-rating");
    ratingBox.hidden = true;
    ratingBox.append(stars(movie));
    body.append(
      el(
        "p",
        "deck-description",
        "Иногда лучший план на вечер — просто хорошее кино. Решать вам.",
      ),
      actions,
      watched,
      ratingBox,
      el("p", "gesture-hint", "← Не интересно · Хочу посмотреть →"),
    );
    const errors = el("div", "card-error");
    errors.setAttribute("role", "status");
    if (failedAction?.movie.movie_id === movie.movie_id) {
      errors.append(
        el("p", "", failedAction.text),
        button("Повторить", () =>
          mutate(movie, failedAction.kind, failedAction.value),
        ),
      );
    }
    body.append(errors);
    card.append(body);
    $("deck").replaceChildren(card);
    setupSwipe(card, movie);
    lock();
  }
  function skip() {
    if (busy || !queue.length) return;
    skipped.add(queue[0].movie_id);
    failedAction = null;
    renderDeck();
    $("deck").focus({ preventScroll: true });
  }
  async function rate(movie, value) {
    return mutate(movie, "rating", value);
  }
  async function mutate(movie, kind, value, focusGrid) {
    if (busy || !ready) return;
    const active = document.activeElement;
    if (active.dataset.rating) focusGrid ??= active.closest(".grid")?.id;
    const focusDeck = $("deck").contains(active);
    const restoreStarFocus = () => {
      if (!focusGrid) return;
      if (document.activeElement !== document.body && document.activeElement !== active)
        return;
      const star = document.querySelector(
        `#${focusGrid} [data-movie-id="${movie.movie_id}"] [data-rating="${value}"]`,
      );
      if (star?.getClientRects().length) star.focus({ preventScroll: true });
    };
    busy = true;
    failedAction = null;
    announce("Сохраняем…");
    // Invalidate reads started before this write, including a previous history restore.
    ++recSeq;
    recAbort?.abort();
    ++historySeq;
    ++watchSeq;
    lock();
    const path =
      kind === "rating"
        ? "/ratings"
        : kind === "remove"
          ? `/watchlist/${movie.movie_id}`
          : "/feedback";
    const payload =
      kind === "rating"
        ? { movie_id: movie.movie_id, rating: value }
        : { movie_id: movie.movie_id, state: kind };
    try {
      await request(path, {
        method: kind === "remove" ? "DELETE" : "POST",
        ...(kind === "remove" ? {} : { body: JSON.stringify(payload) }),
      });
      // Reads may also have started while this write was pending.
      ++historySeq;
      ++watchSeq;
      if (kind === "rating")
        ratings.set(movie.movie_id, { ...movie, user_rating: value });
      if (kind === "remove") {
        watchlist = watchlist.filter((m) => m.movie_id !== movie.movie_id);
        seen.delete(movie.movie_id);
      } else {
        seen.add(movie.movie_id);
        watchlist = watchlist.filter((m) => m.movie_id !== movie.movie_id);
        if (kind === "watchlist") watchlist.unshift(movie);
      }
      queue = queue.filter((m) => m.movie_id !== movie.movie_id);
      busy = false;
      renderLists();
      renderDeck();
      restoreStarFocus();
      if (focusDeck && view === "selection" && choosing) {
        $("deck").focus({ preventScroll: true });
      }
      announce(
        kind === "rating"
          ? "Оценка сохранена"
          : kind === "watchlist"
            ? "Фильм сохранён на вечер"
            : kind === "remove"
              ? "Фильм убран из списка"
              : "Покажем другую историю",
      );
      // A failed list refresh must never turn an already confirmed write into a failed write.
      loadWatchlist().catch(() =>
        announce("Изменения сохранены. Список пока не удалось обновить."),
      );
      if (choosing) loadRecommendations();
    } catch (err) {
      busy = false;
      failedAction = { movie, kind, value, text: errorCopy(err) };
      announce(errorCopy(err));
      // Preserve the exact card and its position; retry repeats the idempotent write.
      if (queue[0]?.movie_id === movie.movie_id) renderDeck();
      else {
        document.querySelectorAll(".movie-card").forEach((card) => {
          if (Number(card.dataset.movieId) !== movie.movie_id) return;
          card.querySelector(".card-error")?.remove();
          const error = el("div", "card-error");
          error.append(
            el("p", "", errorCopy(err)),
            button("Повторить", () => mutate(movie, kind, value, focusGrid)),
          );
          card.append(error);
        });
      }
      lock();
      restoreStarFocus();
    }
  }
  function setupSwipe(card, movie) {
    let origin = null;
    card.addEventListener("dragstart", (event) => event.preventDefault());
    card.addEventListener("pointerdown", (event) => {
      if (
        busy ||
        !event.isPrimary ||
        event.button !== 0 ||
        event.target.closest("button, a, input")
      )
        return;
      origin = { x: event.clientX, y: event.clientY, id: event.pointerId };
      card.setPointerCapture(event.pointerId);
    });
    card.addEventListener("pointerup", (event) => {
      if (!origin || origin.id !== event.pointerId) return;
      const dx = event.clientX - origin.x,
        dy = event.clientY - origin.y;
      origin = null;
      if (Math.abs(dx) >= 80 && Math.abs(dx) >= 1.5 * Math.abs(dy))
        mutate(movie, dx > 0 ? "watchlist" : "dismissed");
    });
    card.addEventListener("pointercancel", () => {
      origin = null;
    });
    card.addEventListener("lostpointercapture", () => {
      origin = null;
    });
  }
  function navigate(next) {
    if (!ready) return;
    view = next;
    for (const id of ["selection", "watchlist", "history"])
      $(id).hidden = id !== view;
    $("search-panel").hidden = false;
    document.querySelectorAll("[data-view]").forEach((node) => {
      if (node.dataset.view === view) node.setAttribute("aria-current", "page");
      else node.removeAttribute("aria-current");
    });
    $("onboarding").hidden = choosing;
    $("recommendations").hidden = !choosing;
    if (view === "watchlist") {
      message($("watchlist-grid"), "Открываем ваши фильмы…");
      loadWatchlist().catch(() =>
        message(
          $("watchlist-grid"),
          "Не удалось открыть сохранённые фильмы.",
          () => navigate("watchlist"),
        ),
      );
    }
    if (view === "history") {
      message($("history-grid"), "Вспоминаем ваши оценки…");
      loadHistory()
        .then((applied) => {
          if (applied) {
            renderLists();
            lock();
          }
        })
        .catch(() =>
          message(
            $("history-grid"),
            "Не удалось открыть оценки. Они сохранены, попробуйте ещё раз.",
            () => navigate("history"),
          ),
        );
    }
  }
  function setupSearch() {
    $("search").addEventListener("input", () => {
      const mine = ++searchSeq;
      clearTimeout(searchTimer);
      searchAbort?.abort();
      const query = $("search").value.trim();
      searchItems = [];
      $("search-grid").replaceChildren();
      if (!ready || query.length < 2) return;
      message($("search-grid"), "Ищем фильм…");
      searchTimer = setTimeout(async () => {
        searchAbort = new AbortController();
        try {
          const data = await request(
            `/movies/search?q=${encodeURIComponent(query)}`,
            { signal: searchAbort.signal },
          );
          if (mine !== searchSeq) return;
          searchItems = data.items;
          grid(
            "search-grid",
            searchItems,
            "search",
            "Ничего не нашлось. Попробуйте другое название.",
          );
          lock();
        } catch (err) {
          if (mine === searchSeq)
            message($("search-grid"), "Поиск временно недоступен.", () =>
              $("search").dispatchEvent(new Event("input")),
            );
        }
      }, 250);
    });
  }
  async function startup() {
    ready = false;
    message($("startup"), "Вспоминаем ваши фильмы…");
    try {
      await loadHistory();
      ready = true;
      choosing = ratings.size >= MIN;
      $("startup").hidden = true;
      renderLists();
      navigate("selection");
      loadOnboarding();
      if (choosing) loadRecommendations();
    } catch (err) {
      message(
        $("startup"),
        "Не удалось открыть ваши оценки. Попробуйте снова — мы продолжим с того же места.",
        startup,
      );
    }
  }
  document
    .querySelectorAll("[data-view]")
    .forEach((node) =>
      node.addEventListener("click", () => navigate(node.dataset.view)),
    );
  $("start-selection").addEventListener("click", () => {
    if (!ready || busy || ratings.size < MIN) return;
    choosing = true;
    navigate("selection");
    loadRecommendations();
  });
  $("deck").addEventListener("keydown", (event) => {
    if (event.target !== $("deck") || busy || $("about").open || !queue.length)
      return;
    if (!["ArrowLeft", "ArrowRight", " "].includes(event.key)) return;
    event.preventDefault();
    if (event.key === " ") skip();
    else
      mutate(queue[0], event.key === "ArrowLeft" ? "dismissed" : "watchlist");
  });
  $("about-open").addEventListener("click", () => $("about").showModal());
  $("about-close").addEventListener("click", () => $("about").close());
  setupSearch();
  startup();
})();
