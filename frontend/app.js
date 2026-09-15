/* ═══════════════════════════════════════════════════════════════
   BlogGraph — Frontend Logic
   Hash-based SPA router: #home | #login | #signup | #app
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = '';

/* ─── Auth session (only the token is kept in the browser; all data lives on the server) ── */
const TOKEN_KEY = 'bloggraph_token';
function getToken()   { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } }
function setToken(t)  { try { localStorage.setItem(TOKEN_KEY, t); } catch {} }
function clearToken() { try { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem('mock_user_id'); } catch {} }

/* fetch wrapper: sends the session token, and returns to login when the session has expired */
async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (options.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json';
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (res.status === 401 && token) {
    clearToken();
    navigate('login');
  }
  return res;
}

async function readError(res) {
  const data = await res.json().catch(() => ({}));
  return data.detail || `Server error ${res.status}`;
}

/* ─── All page elements ────────────────────────────────────────── */
const pages = {
  home:   document.getElementById('page-landing'),
  login:  document.getElementById('page-login'),
  signup: document.getElementById('page-signup'),
  app:    document.getElementById('page-app'),
};

/* ─── Page title map ───────────────────────────────────────────── */
const titles = {
  home:   'BlogGraph — The all-in-one AI workspace for writers',
  login:  'Log In — BlogGraph',
  signup: 'Sign Up — BlogGraph',
  app:    'BlogGraph — Workspace',
};

/* ════════════════════════════════════════════════════════════════
   ROUTER  —  hash-based with full browser history support
   ════════════════════════════════════════════════════════════════ */
function getRoute() {
  const hash = location.hash.replace('#', '').toLowerCase().trim();
  return ['home', 'login', 'signup', 'app'].includes(hash) ? hash : 'home';
}

function navigate(route, pushState = true) {
  let validRoute = ['home', 'login', 'signup', 'app'].includes(route) ? route : 'home';
  // The workspace needs an account; logged-in users skip the auth pages
  if (validRoute === 'app' && !getToken()) validRoute = 'login';
  else if ((validRoute === 'login' || validRoute === 'signup') && getToken()) validRoute = 'app';

  // Push to browser history so the Back button works
  if (pushState) {
    history.pushState({ route: validRoute }, titles[validRoute], `#${validRoute}`);
  }

  // Hide all pages, show the target
  Object.values(pages).forEach(p => { p.hidden = true; });
  pages[validRoute].hidden = false;

  // Update title
  document.title = titles[validRoute];

  // Scroll to top
  window.scrollTo(0, 0);

  // If entering the app page, load user's saved blogs
  if (validRoute === 'app') {
    loadUserBlogs();
  }
}

/* Handle browser back / forward buttons */
window.addEventListener('popstate', (e) => {
  const route = (e.state && e.state.route) ? e.state.route : getRoute();
  navigate(route, false); // false = don't push again
});

/* Intercept all [data-nav] links and clicks on hash links */
document.addEventListener('click', (e) => {
  const link = e.target.closest('[data-nav]');
  if (link) {
    e.preventDefault();
    navigate(link.dataset.nav);
    return;
  }
  // Also handle plain <a href="#route"> links within our pages
  const anchor = e.target.closest('a[href^="#"]');
  if (anchor) {
    const route = anchor.getAttribute('href').replace('#', '');
    if (['home', 'login', 'signup', 'app'].includes(route)) {
      e.preventDefault();
      navigate(route);
    }
  }
});

/* ─── Boot: set initial route ──────────────────────────────────── */
(function boot() {
  // Handle Google OAuth callback: /#google_auth:token=...&email=...&name=...
  const hash = location.hash;
  if (hash.startsWith('#google_auth:')) {
    const params = new URLSearchParams(hash.slice('#google_auth:'.length));
    const token = params.get('token');
    const name  = params.get('name') || '';
    const email = params.get('email') || '';
    const id    = params.get('id') || '';
    if (token) {
      setToken(token);
      localStorage.setItem('mock_user_id', email.split('@')[0] || 'google_user');

      localStorage.setItem('user_name', name);
      localStorage.setItem('user_email', email);
      // Clean up URL and go to app
      history.replaceState({ route: 'app' }, '', '#app');
      navigate('app', false);
      return;
    }
  }

  const initial = getRoute();
  // Replace state (not push) so hitting back from home exits the app
  history.replaceState({ route: initial }, titles[initial], `#${initial}`);
  navigate(initial, false);
})();

/* ════════════════════════════════════════════════════════════════
   AUTH FORMS — Login & Signup UX
   ════════════════════════════════════════════════════════════════ */

/* ── Password visibility toggle ──────────────────────────────── */
function setupEyeToggle(eyeBtnId, inputId) {
  const btn   = document.getElementById(eyeBtnId);
  const input = document.getElementById(inputId);
  if (!btn || !input) return;
  btn.addEventListener('click', () => {
    const show = input.type === 'password';
    input.type = show ? 'text' : 'password';
    btn.title  = show ? 'Hide password' : 'Show password';
    btn.style.color = show ? 'var(--black)' : '';
  });
}

setupEyeToggle('login-eye',  'login-password');
setupEyeToggle('signup-eye', 'signup-password');

/* ── Password strength meter ─────────────────────────────────── */
const signupPasswordInput = document.getElementById('signup-password');
const strengthWrapper     = document.getElementById('signup-strength');
const bars  = [
  document.getElementById('sb1'),
  document.getElementById('sb2'),
  document.getElementById('sb3'),
  document.getElementById('sb4'),
];
const strengthLabel = document.getElementById('strength-label');

const strengthClasses = ['active-weak', 'active-fair', 'active-good', 'active-strong'];
const strengthTexts   = ['Weak', 'Fair', 'Good', 'Strong'];

function calcStrength(pw) {
  let score = 0;
  if (pw.length >= 8)  score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  return score; // 0-4
}

if (signupPasswordInput) {
  signupPasswordInput.addEventListener('input', () => {
    const pw = signupPasswordInput.value;
    if (!pw) { strengthWrapper.hidden = true; return; }
    strengthWrapper.hidden = false;
    const score = calcStrength(pw);
    bars.forEach((bar, i) => {
      bar.className = 'strength-bar';
      if (i < score) bar.classList.add(strengthClasses[score - 1]);
    });
    strengthLabel.textContent = strengthTexts[score - 1] || 'Weak';
    strengthLabel.style.color = ['var(--red)', 'var(--orange)', 'var(--amber)', 'var(--green)'][score - 1] || 'var(--red)';
  });
}

/* ── Login form submit ────────────────────────────────────────── */
const loginForm      = document.getElementById('login-form');
const loginSubmitBtn = document.getElementById('login-submit-btn');
const loginBtnText   = document.getElementById('login-btn-text');
const loginSpinner   = document.getElementById('login-spinner');

if (loginForm) {
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('login-email').value.trim();
    const pass  = document.getElementById('login-password').value;

    // Basic validation
    let valid = true;
    if (!email || !email.includes('@')) {
      document.getElementById('login-email').classList.add('input-error');
      valid = false;
    } else {
      document.getElementById('login-email').classList.remove('input-error');
    }
    if (!pass) {
      document.getElementById('login-password').classList.add('input-error');
      valid = false;
    } else {
      document.getElementById('login-password').classList.remove('input-error');
    }
    if (!valid) return;

    loginSubmitBtn.disabled = true;
    loginBtnText.textContent = 'Signing in…';
    loginSpinner.hidden = false;
    showFormError(loginForm, '');

    try {
      const res = await apiFetch('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password: pass, remember: !!document.getElementById('login-remember')?.checked }),
      });
      if (!res.ok) throw new Error(await readError(res));
      onAuthSuccess(await res.json()); // on success → go to app
    } catch (err) {
      showFormError(loginForm, err.message || 'Could not sign in. Please try again.');
    } finally {
      loginSpinner.hidden = true;
      loginBtnText.textContent = 'Sign In';
      loginSubmitBtn.disabled = false;
    }
  });

  // Clear error on input
  ['login-email', 'login-password'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => {
      document.getElementById(id).classList.remove('input-error');
    });
  });
}

/* Password reset is not available yet */
document.querySelector('#page-login .auth-label-row .auth-link-small')?.addEventListener('click', (e) => {
  e.preventDefault();
  showFormError(loginForm, 'Password reset is not available yet.');
});

/* ── Signup form submit ───────────────────────────────────────── */
const signupForm      = document.getElementById('signup-form');
const signupSubmitBtn = document.getElementById('signup-submit-btn');
const signupBtnText   = document.getElementById('signup-btn-text');
const signupSpinner   = document.getElementById('signup-spinner');

if (signupForm) {
  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email    = document.getElementById('signup-email').value.trim();
    const password = document.getElementById('signup-password').value;
    const terms    = document.getElementById('signup-terms').checked;

    let valid = true;
    if (!email || !email.includes('@')) {
      document.getElementById('signup-email').classList.add('input-error');
      valid = false;
    } else {
      document.getElementById('signup-email').classList.remove('input-error');
    }
    if (password.length < 8) {
      document.getElementById('signup-password').classList.add('input-error');
      valid = false;
    } else {
      document.getElementById('signup-password').classList.remove('input-error');
    }
    if (!terms) {
      alert('Please agree to the Terms of Service to continue.');
      valid = false;
    }
    if (!valid) return;

    signupSubmitBtn.disabled = true;
    signupBtnText.textContent = 'Creating account…';
    signupSpinner.hidden = false;
    showFormError(signupForm, '');

    const name = [document.getElementById('signup-fname')?.value, document.getElementById('signup-lname')?.value]
      .map(s => (s || '').trim()).filter(Boolean).join(' ');

    try {
      const res = await apiFetch('/auth/signup', {
        method: 'POST',
        body: JSON.stringify({ name, email, password }),
      });
      if (!res.ok) throw new Error(await readError(res));
      onAuthSuccess(await res.json()); // on success → go to app
    } catch (err) {
      showFormError(signupForm, err.message || 'Could not create your account. Please try again.');
    } finally {
      signupSpinner.hidden = true;
      signupBtnText.textContent = 'Create Free Account';
      signupSubmitBtn.disabled = false;
    }
  });

  ['signup-email', 'signup-password'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => {
      document.getElementById(id).classList.remove('input-error');
    });
  });
}


/* ── Auth helpers ─────────────────────────────────────────────── */
function showFormError(form, message) {
  if (!form) return;
  let box = form.querySelector('.auth-form-error');
  if (!box) {
    box = document.createElement('div');
    box.className = 'auth-form-error';
    box.setAttribute('role', 'alert');
    form.prepend(box);
  }
  box.textContent = message;
  box.hidden = !message;
}

function onAuthSuccess(data) {
  setToken(data.token);
  setAccountInfo(data.user);
  [loginForm, signupForm].forEach(f => { f?.reset(); showFormError(f, ''); });
  if (strengthWrapper) strengthWrapper.hidden = true;
  navigate('app');
}

function setAccountInfo(user) {
  const btn = document.getElementById('logout-btn');
  if (btn && user) btn.title = `Log out (${user.email})`;
}

async function logout() {
  if (!confirm('Log out of BlogGraph?')) return;
  try { await apiFetch('/auth/logout', { method: 'POST' }); } catch {}
  clearToken();
  const list = document.getElementById('saved-blogs-list');
  if (list) list.innerHTML = '';
  newBlog();
  navigate('home');
}

document.getElementById('logout-btn')?.addEventListener('click', logout);


/* ════════════════════════════════════════════════════════════════
   BLOG GENERATOR  (App page)
   ════════════════════════════════════════════════════════════════ */
const form         = document.getElementById('generate-form');
const topicInput   = document.getElementById('topic-input');
const dateInput    = document.getElementById('date-input');
const generateBtn  = document.getElementById('generate-btn');
const btnLabel     = document.getElementById('btn-label');

const emptyState   = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const errorState   = document.getElementById('error-state');
const blogPreview  = document.getElementById('blog-preview');
const errorMessage = document.getElementById('error-message');
const retryBtn     = document.getElementById('retry-btn');

const modeValue    = document.getElementById('mode-value');
const modeDot      = document.getElementById('mode-dot');
const appModeDot   = document.getElementById('app-mode-dot');
const appModeText  = document.getElementById('app-mode-text');

const statSections       = document.getElementById('stat-sections');
const statWords          = document.getElementById('stat-words');
const statsRow           = document.getElementById('stats-row');
const resultModeBadge2   = document.getElementById('result-mode-badge-2');
const resultSectionsText = document.getElementById('result-sections-text');
const blogContent        = document.getElementById('blog-content');
const copyBtn            = document.getElementById('copy-btn');
const downloadBtn        = document.getElementById('download-btn');

const loadingSteps = [
  document.getElementById('step-routing'),
  document.getElementById('step-research'),
  document.getElementById('step-planning'),
  document.getElementById('step-writing'),
  document.getElementById('step-images'),
];

let stepTimer = null;
let currentMarkdown = '';
let currentBlogId = null; // id of the saved blog currently on screen (null for unsaved/new)

/* ── Init date ─────────────────────────────────────────────────── */
if (dateInput) dateInput.value = new Date().toISOString().split('T')[0];

/* ── Show/hide states ──────────────────────────────────────────── */
function showPanel(which) {
  if (emptyState)   emptyState.hidden   = which !== 'empty';
  if (loadingState) loadingState.hidden = which !== 'loading';
  if (errorState)   errorState.hidden   = which !== 'error';
  if (blogPreview)  blogPreview.hidden  = which !== 'result';
}

/* ── Loading steps animation ───────────────────────────────────── */
function startLoadingAnimation() {
  loadingSteps.forEach(s => s && (s.className = 'app-step'));
  if (loadingSteps[0]) loadingSteps[0].classList.add('active');
  const intervals = [8000, 16000, 28000, 48000];
  stepTimer = setTimeout(function advance(i) {
    if (i >= loadingSteps.length) return;
    if (loadingSteps[i - 1]) { loadingSteps[i-1].classList.remove('active'); loadingSteps[i-1].classList.add('done'); }
    if (loadingSteps[i]) loadingSteps[i].classList.add('active');
    if (i + 1 < loadingSteps.length) stepTimer = setTimeout(() => advance(i + 1), intervals[i] || 10000);
  }, intervals[0], 1);
}

function stopLoadingAnimation() {
  clearTimeout(stepTimer);
  loadingSteps.forEach(s => s && (s.classList.remove('active'), s.classList.add('done')));
}

/* ── Mode indicator ────────────────────────────────────────────── */
function setModeIndicator(mode) {
  const labels = {
    closed_book: { text: 'Closed Book (No Search)', cls: 'green',  short: 'Closed Book' },
    hybrid:      { text: 'Hybrid (Targeted Search)', cls: 'amber',  short: 'Hybrid' },
    open_book:   { text: 'Open Book (Full Research)', cls: 'purple', short: 'Open Book' },
  };
  const info = labels[mode] || { text: mode, cls: 'green', short: mode };
  if (modeValue)   modeValue.textContent   = info.text;
  if (modeDot)     modeDot.className       = 'app-mode-indicator-dot ' + info.cls;
  if (appModeDot)  appModeDot.className    = 'app-mode-dot ' + info.cls;
  if (appModeText) appModeText.textContent = 'Research: ' + info.short;
}

/* ── Markdown renderer ─────────────────────────────────────────── */
function renderMarkdown(md) {
  if (!blogContent) return;
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
    const html = marked.parse(md);
    // LLM / web-sourced markdown can contain raw HTML, so sanitize before inserting
    blogContent.innerHTML = typeof DOMPurify !== 'undefined'
      ? DOMPurify.sanitize(html)
      : `<pre style="white-space:pre-wrap">${escapeHtml(md)}</pre>`;
  } else {
    blogContent.innerHTML = `<pre style="white-space:pre-wrap">${escapeHtml(md)}</pre>`;
  }
}

function escapeHtml(str) {
  return String(str ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ── Generate handler ──────────────────────────────────────────── */
async function handleGenerate(e) {
  e.preventDefault();
  if (!topicInput) return;

  const topic = topicInput.value.trim();
  if (!topic) {
    topicInput.focus();
    topicInput.style.borderColor = 'var(--red)';
    topicInput.style.boxShadow   = '0 0 0 3px rgba(239,68,68,.12)';
    setTimeout(() => { topicInput.style.borderColor = ''; topicInput.style.boxShadow = ''; }, 2500);
    return;
  }

  const as_of = dateInput ? dateInput.value || new Date().toISOString().split('T')[0] : new Date().toISOString().split('T')[0];

  generateBtn.disabled  = true;
  if (btnLabel) btnLabel.textContent = 'Generating…';
  showPanel('loading');
  startLoadingAnimation();
  addChatMsg('agent', 'Generating your blog post about: <strong>' + escapeHtml(topic) + '</strong>. This may take 30–90 seconds…');

  try {
    const response = await apiFetch('/generate', {
      method: 'POST',
      body: JSON.stringify({ topic, as_of }),
    });

    if (!response.ok) throw new Error(await readError(response));

    const data = await response.json();
    stopLoadingAnimation();
    currentMarkdown = data.markdown || '';

    setModeIndicator(data.mode);

    const wordCount = currentMarkdown.trim().split(/\s+/).length;
    if (statSections) statSections.textContent = data.sections_count;
    if (statWords)    statWords.textContent    = wordCount.toLocaleString();
    if (statsRow)     statsRow.hidden          = false;

    const modeShort = (data.mode || 'unknown').replace(/_/g, ' ');
    if (resultModeBadge2)   resultModeBadge2.textContent   = modeShort;
    if (resultSectionsText) resultSectionsText.textContent = `${data.sections_count} section${data.sections_count !== 1 ? 's' : ''}`;

    renderMarkdown(currentMarkdown);
    showPanel('result');

    addChatMsg('agent', `✓ Done! Generated <strong>${data.sections_count} sections</strong> (${wordCount.toLocaleString()} words) in <em>${modeShort}</em> mode.`);

    // The server saves the blog to your account as part of generation
    currentBlogId = data.blog?.id ?? null; // lets delete clear the workspace if this blog is removed
    loadUserBlogs(); // Refresh sidebar list

  } catch (err) {
    stopLoadingAnimation();
    if (errorMessage) errorMessage.textContent = err.message || 'An unexpected error occurred.';
    showPanel('error');
    addChatMsg('agent', '⚠ Generation failed: ' + (err.message || 'Please try again.'));
  } finally {
    generateBtn.disabled = false;
    if (btnLabel) btnLabel.textContent = 'Generate Blog';
  }
}

/* ── Chat ──────────────────────────────────────────────────────── */
const chatMessages = document.getElementById('chat-messages');

function addChatMsg(role, html) {
  if (!chatMessages) return;
  const div = document.createElement('div');
  div.className = role === 'user' ? 'app-chat-msg app-chat-msg-user' : 'app-chat-msg';
  const p = document.createElement('p');
  p.className = 'app-chat-text';
  p.innerHTML = html;
  div.appendChild(p);
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/* Social button login — redirect to backend Google OAuth */
document.getElementById('login-google-btn')?.addEventListener('click', () => { window.location.href = '/auth/google'; });
document.getElementById('login-github-btn')?.addEventListener('click',  () => { localStorage.setItem('mock_user_id', 'github_user'); navigate('app'); });

document.getElementById('chat-send-btn')?.addEventListener('click', handleChatSend);
document.getElementById('chat-input')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleChatSend(); }
});

function handleChatSend() {
  const input = document.getElementById('chat-input');
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;
  addChatMsg('user', escapeHtml(text));
  input.value = '';
  setTimeout(() => addChatMsg('agent', 'Thanks for your message! Configure a topic in the left panel and click <strong>Generate Blog</strong> to create content.'), 600);
}

/* ── Copy ──────────────────────────────────────────────────────── */
async function handleCopy() {
  if (!currentMarkdown) return;
  try {
    await navigator.clipboard.writeText(currentMarkdown);
    if (!copyBtn) return;
    const orig = copyBtn.innerHTML;
    copyBtn.innerHTML = '<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg> Copied!';
    setTimeout(() => { copyBtn.innerHTML = orig; }, 2200);
  } catch { if (copyBtn) copyBtn.textContent = 'Failed'; }
}

/* ── Download ──────────────────────────────────────────────────── */
function handleDownload() {
  if (!currentMarkdown) return;
  // Name the file after the blog's H1 title (falls back to the topic box)
  const h1 = (currentMarkdown.match(/^#\s+(.+)$/m) || [])[1];
  const slug = (h1 || topicInput?.value || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 50) || 'blog';
  const blob = new Blob([currentMarkdown], { type: 'text/markdown;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `${slug}.md`;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

/* ── Event listeners ───────────────────────────────────────────── */
form?.addEventListener('submit', handleGenerate);
retryBtn?.addEventListener('click', () => showPanel('empty'));
copyBtn?.addEventListener('click', handleCopy);
downloadBtn?.addEventListener('click', handleDownload);
topicInput?.addEventListener('input', () => {
  topicInput.style.borderColor = '';
  topicInput.style.boxShadow   = '';
});

/* ── Init app panel ────────────────────────────────────────────── */
showPanel('empty');

/* ── New Blog: reset everything for a fresh session ────────────── */
function newBlog() {
  // Clear current content
  currentMarkdown = '';
  currentBlogId = null;

  // Reset form fields
  if (topicInput) topicInput.value = '';
  if (dateInput)  dateInput.value = new Date().toISOString().split('T')[0];

  // Reset mode indicators
  setModeIndicator('');

  // Hide stats row
  if (statsRow) statsRow.hidden = true;

  // Clear active state on sidebar items
  document.querySelectorAll('.app-sidebar-ws-item').forEach(el => el.classList.remove('app-sidebar-ws-active'));

  // Show empty state
  showPanel('empty');

  // Focus the topic input so the user can type right away
  if (topicInput) topicInput.focus();
}

document.getElementById('new-blog-btn')?.addEventListener('click', newBlog);


/* ── Database: Load & Display Saved Blogs ──────────────────────── */
async function loadUserBlogs() {
  const listContainer = document.getElementById('saved-blogs-list');
  if (!listContainer || !getToken()) return;

  try {
    // Blogs are stored on the server under your account, so they follow you to any browser or device
    const response = await apiFetch('/blogs');
    if (response.status === 401) return; // apiFetch already sent the user to login
    if (!response.ok) throw new Error('Failed to fetch');
    const data = await response.json();
    const blogs = data.blogs || [];

    listContainer.innerHTML = '';
    
    if (blogs.length === 0) {
      listContainer.innerHTML = '<div style="padding: 4px 16px; font-size: 12px; color: var(--gray-400);">No blogs saved yet.</div>';
      return;
    }

    blogs.forEach(blog => {
      // Row = clickable title + "⋯" menu button (shown on hover)
      const row = document.createElement('div');
      row.className = 'app-sidebar-ws-row';
      row.dataset.blogId = blog.id;

      const a = document.createElement('a');
      a.href = '#';
      a.className = 'app-sidebar-ws-item';
      if (blog.id === currentBlogId) a.classList.add('app-sidebar-ws-active');

      // Limit title length for sidebar
      let title = blog.blog_title || blog.topic || 'Untitled Blog';
      a.title = title;
      if (title.length > 28) title = title.substring(0, 26) + '…';

      a.textContent = `📄 ${title}`;

      a.addEventListener('click', (e) => {
        e.preventDefault();

        // Remove active class from all, add to this
        document.querySelectorAll('.app-sidebar-ws-item').forEach(el => el.classList.remove('app-sidebar-ws-active'));
        a.classList.add('app-sidebar-ws-active');

        viewSavedBlog(blog);
      });

      const menuBtn = document.createElement('button');
      menuBtn.type = 'button';
      menuBtn.className = 'app-sidebar-ws-menu-btn';
      menuBtn.title = 'More options';
      menuBtn.setAttribute('aria-label', 'More options');
      menuBtn.setAttribute('aria-haspopup', 'menu');
      menuBtn.textContent = '⋯';
      menuBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        toggleBlogMenu(row, blog);
      });

      row.appendChild(a);
      row.appendChild(menuBtn);
      listContainer.appendChild(row);
    });

  } catch (err) {
    console.error("Error loading user blogs:", err);
    listContainer.innerHTML = '<div style="padding: 4px 16px; font-size: 12px; color: var(--red);">Could not load saved blogs. Is the server running?</div>';
  }
}

/* ── Saved blog "⋯" menu: Delete ───────────────────────────────── */
function closeBlogMenus() {
  document.querySelectorAll('.app-sidebar-ws-menu').forEach(m => m.remove());
  document.querySelectorAll('.app-sidebar-ws-row.menu-open').forEach(r => r.classList.remove('menu-open'));
}

function toggleBlogMenu(row, blog) {
  const wasOpen = row.classList.contains('menu-open');
  closeBlogMenus();
  if (wasOpen) return;

  row.classList.add('menu-open');
  const menu = document.createElement('div');
  menu.className = 'app-sidebar-ws-menu';
  menu.setAttribute('role', 'menu');

  const del = document.createElement('button');
  del.type = 'button';
  del.className = 'app-sidebar-ws-menu-item app-sidebar-ws-menu-danger';
  del.setAttribute('role', 'menuitem');
  del.innerHTML = '<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14" aria-hidden="true"><path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/></svg> Delete';
  del.addEventListener('click', (e) => {
    e.stopPropagation();
    closeBlogMenus();
    deleteSavedBlog(blog, row);
  });

  menu.appendChild(del);
  row.appendChild(menu);
}

// Close the menu when clicking elsewhere or pressing Escape
document.addEventListener('click', (e) => {
  if (!e.target.closest('.app-sidebar-ws-menu')) closeBlogMenus();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeBlogMenus(); });

async function deleteSavedBlog(blog, row) {
  const name = blog.blog_title || blog.topic || 'this blog';
  if (!confirm(`Delete "${name}"?\n\nThis permanently removes it from your saved blogs.`)) return;

  row.classList.add('deleting');
  try {
    const res = await apiFetch(`/blogs/${blog.id}`, { method: 'DELETE' });
    // 404 means it is already gone, which is the outcome we want
    if (!res.ok && res.status !== 404) throw new Error(await readError(res));

    // If the deleted blog is the one on screen, clear the workspace
    if (currentBlogId === blog.id) newBlog();

    addChatMsg('agent', `Deleted blog: <strong>${escapeHtml(name)}</strong>`);
    loadUserBlogs();
  } catch (err) {
    row.classList.remove('deleting');
    addChatMsg('agent', '⚠ Could not delete blog: ' + escapeHtml(err.message || 'Please try again.'));
  }
}

function viewSavedBlog(blog) {
  // Populate UI with the saved blog data
  currentBlogId = blog.id;
  currentMarkdown = blog.markdown || '';
  
  if (topicInput) topicInput.value = blog.topic;
  
  setModeIndicator(blog.mode);
  
  const wordCount = currentMarkdown.trim().split(/\s+/).length;
  if (statSections) statSections.textContent = blog.sections_count;
  if (statWords)    statWords.textContent    = wordCount.toLocaleString();
  if (statsRow)     statsRow.hidden          = false;
  
  const modeShort = (blog.mode || 'unknown').replace(/_/g, ' ');
  if (resultModeBadge2)   resultModeBadge2.textContent   = modeShort;
  if (resultSectionsText) resultSectionsText.textContent = `${blog.sections_count} section${blog.sections_count !== 1 ? 's' : ''}`;
  
  renderMarkdown(currentMarkdown);
  showPanel('result');
  
  addChatMsg('agent', `Loaded saved blog: <strong>${escapeHtml(blog.blog_title || blog.topic)}</strong>`);
}

/* ── Validate a stored session on page load (an expired one returns to login) ── */
if (getToken()) {
  apiFetch('/auth/me')
    .then(async res => { if (res.ok) setAccountInfo((await res.json()).user); })
    .catch(() => {});
}
