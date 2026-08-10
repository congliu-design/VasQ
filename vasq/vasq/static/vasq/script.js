document.addEventListener("DOMContentLoaded", function() {
    const messageInput = document.getElementById('message-input');
    const messagesContainer = document.getElementById('messages');
    const form = document.getElementById("chat-form");
    const fileInput = document.getElementById("file-upload");
    const sendButton = document.getElementById("send-message");
    const queueButton = document.getElementById("run-queue");
    let queueRunning = false;
    let stopQueueRequested = false;

    // Export chat as PDF
    document.getElementById('export-pdf').addEventListener('click', function() {
        const element = document.getElementById('messages-container');
        const opt = {
            margin: [0, 0, 0, 0],
            filename: 'chat_history.pdf',
            image: { type: 'jpeg', quality: 1.0 },
            html2canvas: {
                scale: 2,
                windowWidth: element.scrollWidth,
                windowHeight: element.scrollHeight,
            },
            jsPDF: {
                unit: 'px',
                format: [element.scrollWidth, element.scrollHeight],
                orientation: 'portrait',
            }
        };
        html2pdf().set(opt).from(element).save();
    });

    // Export chat as text file
    document.getElementById('export-chat').addEventListener('click', function() {
        let chatHistory = '';
        const messages = messagesContainer.querySelectorAll('.user-message, .system-message');
        messages.forEach(function(message) {
            if (message.classList.contains('user-message')) {
                chatHistory += `User: ${message.innerText}\n`;
            } else if (message.classList.contains('system-message')) {
                chatHistory += `VasQ: ${message.innerText}\n`;
            }
        });
        const blob = new Blob([chatHistory], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat_history.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // Scroll to bottom
    function scrollToBottom() {
        const lastMessage = messagesContainer.lastElementChild;
        if (lastMessage) {
            lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }
    scrollToBottom();

    // Text display
    function typeWriter(text, element, callback) {
        let i = 0;
        const fragment = document.createDocumentFragment();
    
        function appendText(value, parent = fragment) {
            parent.appendChild(document.createTextNode(value));
        }
    
        function appendDelimited(marker, tagName, nestedTagName = null) {
            if (!text.startsWith(marker, i)) return false;
    
            const contentStart = i + marker.length;
            const contentEnd = text.indexOf(marker, contentStart);
            if (contentEnd === -1) return false;
    
            const outerElement = document.createElement(tagName);
    
            if (nestedTagName) {
                const innerElement = document.createElement(nestedTagName);
                innerElement.textContent = text.slice(contentStart, contentEnd);
                outerElement.appendChild(innerElement);
            } else {
                outerElement.textContent = text.slice(contentStart, contentEnd);
            }
    
            fragment.appendChild(outerElement);
            i = contentEnd + marker.length;
            return true;
        }

        function safeUrl(rawUrl, allowMailto = false) {
            try {
                const parsed = new URL(rawUrl, window.location.href);
                const allowedProtocols = allowMailto
                    ? ['http:', 'https:', 'mailto:']
                    : ['http:', 'https:'];
    
                return allowedProtocols.includes(parsed.protocol)
                    ? parsed.href
                    : null;
            } catch (error) {
                return null;
            }
        }

        while (i < text.length) {
            const atLineStart =
                i === 0 || text.charAt(i - 1) === '\n';
    
            // Code block：必须放在 inline code 前面
            if (text.startsWith('```', i)) {
                const contentStart = i + 3;
                const contentEnd = text.indexOf('```', contentStart);
    
                if (contentEnd !== -1) {
                    let codeText = text.slice(
                        contentStart,
                        contentEnd
                    );
    
                    codeText = codeText.replace(
                        /^[A-Za-z0-9_+-]+\n/,
                        ''
                    );
    
                    const preElement =
                        document.createElement('pre');
                    const codeElement =
                        document.createElement('code');
    
                    codeElement.textContent = codeText;
                    preElement.appendChild(codeElement);
                    fragment.appendChild(preElement);
    
                    i = contentEnd + 3;
                    continue;
                }
            }

            // Bold italic
            if (
                appendDelimited(
                    '***',
                    'strong',
                    'em'
                )
            ) {
                continue;
            }
    
            // Bold
            if (appendDelimited('**', 'strong')) {
                continue;
            }
    
            // Strikethrough
            if (appendDelimited('~~', 'del')) {
                continue;
            }
    
            // Inline code
            if (appendDelimited('`', 'code')) {
                continue;
            }
    
            // Italic
            if (appendDelimited('*', 'em')) {
                continue;
            }
    
            // H3
            if (
                atLineStart &&
                text.startsWith('###', i)
            ) {
                const end = text.indexOf('\n', i);
                const contentEnd =
                    end === -1 ? text.length : end;
    
                const heading =
                    document.createElement('h3');
    
                heading.textContent = text
                    .slice(i + 3, contentEnd)
                    .trim();
    
                fragment.appendChild(heading);
                i = contentEnd;
                continue;
            }
    
            // H2
            if (
                atLineStart &&
                text.startsWith('##', i)
            ) {
                const end = text.indexOf('\n', i);
                const contentEnd =
                    end === -1 ? text.length : end;
    
                const heading =
                    document.createElement('h2');
    
                heading.textContent = text
                    .slice(i + 2, contentEnd)
                    .trim();
    
                fragment.appendChild(heading);
                i = contentEnd;
                continue;
            }
    
            // H1
            if (
                atLineStart &&
                text.startsWith('#', i)
            ) {
                const end = text.indexOf('\n', i);
                const contentEnd =
                    end === -1 ? text.length : end;
    
                const heading =
                    document.createElement('h1');
    
                heading.textContent = text
                    .slice(i + 1, contentEnd)
                    .trim();
    
                fragment.appendChild(heading);
                i = contentEnd;
                continue;
            }
    
            // Blockquote
            if (
                atLineStart &&
                text.startsWith('>', i)
            ) {
                const end = text.indexOf('\n', i);
                const contentEnd =
                    end === -1 ? text.length : end;
    
                const quote =
                    document.createElement('blockquote');
    
                quote.textContent = text
                    .slice(i + 1, contentEnd)
                    .trim();
    
                fragment.appendChild(quote);
                i = contentEnd;
                continue;
            }
    
            // Image
            const imageMatch = text
                .slice(i)
                .match(
                    /^!\[([^\]]*)\]\(([^)]+)\)/
                );
    
            if (imageMatch) {
                const imageUrl =
                    safeUrl(imageMatch[2]);
    
                if (imageUrl) {
                    const image =
                        document.createElement('img');
    
                    image.alt = imageMatch[1];
                    image.src = imageUrl;
                    image.style.maxWidth = '100%';
    
                    fragment.appendChild(image);
                } else {
                    appendText(imageMatch[1]);
                }
    
                i += imageMatch[0].length;
                continue;
            }
    
            // Link
            const linkMatch = text
                .slice(i)
                .match(
                    /^\[([^\]]+)\]\(([^)]+)\)/
                );
    
            if (linkMatch) {
                const linkUrl =
                    safeUrl(linkMatch[2], true);
    
                if (linkUrl) {
                    const link =
                        document.createElement('a');
    
                    link.textContent = linkMatch[1];
                    link.href = linkUrl;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
    
                    fragment.appendChild(link);
                } else {
                    appendText(linkMatch[1]);
                }
    
                i += linkMatch[0].length;
                continue;
            }
    
            // Horizontal rule
            if (
                atLineStart &&
                text.startsWith('---', i)
            ) {
                fragment.appendChild(
                    document.createElement('hr')
                );
    
                i += 3;
                continue;
            }
    
            // Line break
            if (text.charAt(i) === '\n') {
                fragment.appendChild(
                    document.createElement('br')
                );
    
                i += 1;
                continue;
            }
    
            // 一次添加一段普通文本，不再逐字添加
            const specialCharacters = new Set([
                '*',
                '`',
                '~',
                '>',
                '#',
                '[',
                '!',
                '\n',
                '-'
            ]);
    
            let end = i + 1;
    
            while (
                end < text.length &&
                !specialCharacters.has(
                    text.charAt(end)
                )
            ) {
                end += 1;
            }
    
            appendText(text.slice(i, end));
            i = end;
        }
    
        // 所有内容一次性添加到页面
        element.appendChild(fragment);
        scrollToBottom();
    
        if (callback) {
            callback();
        }
    }

    // Add messages to chat
    function addChatMessage(messageText, isUser) {
        const messageElement = document.createElement('div');
        messageElement.classList.add(isUser ? 'user-message' : 'system-message');
        messagesContainer.appendChild(messageElement);

        if (isTextMessage(messageText)) {
            typeWriter(messageText, messageElement, () => {
                scrollToBottom();
            });
        } else {
            messageElement.textContent = messageText;
            scrollToBottom();
        }
    }
    
    function addTemporaryMessage(messageText) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('system-message', 'thinking-message');
        messageElement.textContent = messageText;
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
        return messageElement;
    }

    function addQueueStatus(messageText) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('system-message', 'queue-status-message');
        messageElement.textContent = messageText;
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
        return messageElement;
    }

    // Identify message type
    function isTextMessage(message) {
        return typeof message === 'string';
    }

    function parseQuestionQueue(rawText) {
        return rawText
            .split(/\r?\n/)
            .map(function(line) {
                return line
                    .trim()
                    .replace(/^(?:\d+[.)]|[-*\u2022])\s+/, '')
                    .trim();
            })
            .filter(Boolean);
    }

    function setQueueControls(isRunning) {
        queueRunning = isRunning;
        messageInput.disabled = isRunning;
        sendButton.disabled = isRunning;

        if (isRunning) {
            queueButton.textContent = 'Stop after current';
            queueButton.disabled = false;
        } else {
            queueButton.textContent = 'Run Queue';
            queueButton.disabled = false;
        }
    }

    function renderGraph(graphJsonValue) {
        if (!graphJsonValue) return;

        try {
            const graphJson = typeof graphJsonValue === 'string'
                ? JSON.parse(graphJsonValue)
                : graphJsonValue;
            const graphDiv = document.createElement('div');
            graphDiv.className = 'vasq-plot-card';
            graphDiv.setAttribute(
                'aria-label',
                'VasQ gene expression across brain regions'
            );
            messagesContainer.appendChild(graphDiv);
            Plotly.react(
                graphDiv,
                graphJson.data,
                graphJson.layout,
                {
                    responsive: true,
                    displaylogo: false,
                    scrollZoom: false,
                    modeBarButtonsToRemove: [
                        'select2d',
                        'lasso2d',
                        'autoScale2d'
                    ]
                }
            );
            scrollToBottom();
        } catch (error) {
            console.error('Could not render graph:', error);
            addChatMessage('The answer was returned, but its graph could not be rendered.', false);
        }
    }

    async function sendQuestion(message, options = {}) {
        const queueIndex = options.queueIndex || null;
        const queueTotal = options.queueTotal || null;
        const resetHistory = options.resetHistory === true;
        const prefix = queueIndex && queueTotal
            ? `[${queueIndex}/${queueTotal}] `
            : '';

        addChatMessage(prefix + message, true);
        const thinkingMessage = addTemporaryMessage(
            queueIndex && queueTotal
                ? `Running question ${queueIndex} of ${queueTotal}...`
                : 'Thinking...'
        );

        try {
            const response = await fetch(chatUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    message: message,
                    reset_history: resetHistory
                })
            });

            const data = await response.json().catch(function() {
                return {};
            });

            if (!response.ok) {
                throw new Error(
                    data.response || `Server returned HTTP ${response.status}`
                );
            }

            if (data.response) {
                addChatMessage(data.response, false);
            } else {
                addChatMessage('I could not generate a response.', false);
            }

            renderGraph(data.graph_json);
            return true;
        } catch (error) {
            console.error('Error:', error);
            addChatMessage(
                `${prefix}Failed: ${error.message || 'Could not send this question.'}`,
                false
            );
            return false;
        } finally {
            thinkingMessage.remove();
            scrollToBottom();
        }
    }

    // Handle form submission
    if (
        form &&
        messageInput &&
        messagesContainer &&
        sendButton &&
        queueButton
    ) {
        form.onsubmit = async function(e) {
            e.preventDefault();
            if (queueRunning) return;

            const message = messageInput.value.trim();
            if (!message) return;

            messageInput.value = '';
            sendButton.disabled = true;
            queueButton.disabled = true;
            await sendQuestion(message, { resetHistory: false });
            sendButton.disabled = false;
            queueButton.disabled = false;
            messageInput.focus();
        };

        queueButton.addEventListener('click', async function() {
            if (queueRunning) {
                stopQueueRequested = true;
                queueButton.textContent = 'Stopping...';
                queueButton.disabled = true;
                return;
            }

            const questions = parseQuestionQueue(messageInput.value);
            if (!questions.length) {
                addQueueStatus('Paste one question per line, then click Run Queue.');
                return;
            }

            if (questions.length > 50) {
                addQueueStatus('A queue can contain at most 50 questions.');
                return;
            }

            messageInput.value = '';
            stopQueueRequested = false;
            setQueueControls(true);
            addQueueStatus(`Queue started: ${questions.length} question(s).`);

            let completed = 0;
            let failed = 0;

            for (let index = 0; index < questions.length; index += 1) {
                if (stopQueueRequested) break;

                const succeeded = await sendQuestion(questions[index], {
                    queueIndex: index + 1,
                    queueTotal: questions.length,
                    resetHistory: true
                });

                if (succeeded) {
                    completed += 1;
                } else {
                    failed += 1;
                }
            }

            if (stopQueueRequested) {
                addQueueStatus(
                    `Queue stopped. Completed: ${completed}; failed: ${failed}.`
                );
            } else {
                addQueueStatus(
                    `Queue finished. Completed: ${completed}; failed: ${failed}.`
                );
            }

            stopQueueRequested = false;
            setQueueControls(false);
            messageInput.focus();
        });
    }
    
});
