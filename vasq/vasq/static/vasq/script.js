document.addEventListener("DOMContentLoaded", function() {
    const messageInput = document.getElementById('message-input');
    const messagesContainer = document.getElementById('messages');
    const form = document.getElementById("chat-form");
    const fileInput = document.getElementById("file-upload");
    const sendButton = document.getElementById("send-message");
    const stopButton = document.getElementById("stop-request");
    const queueButton = document.getElementById("run-queue");
    let queueRunning = false;
    let stopQueueRequested = false;
    let activeRequest = null;

    function createChatId() {
        if (
            window.crypto &&
            typeof window.crypto.randomUUID === 'function'
        ) {
            return window.crypto.randomUUID();
        }
    
        const bytes = new Uint8Array(16);
        window.crypto.getRandomValues(bytes);
    
        // UUID version 4
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
    
        const hex = Array.from(
            bytes,
            value => value.toString(16).padStart(2, '0')
        );
    
        return [
            hex.slice(0, 4).join(''),
            hex.slice(4, 6).join(''),
            hex.slice(6, 8).join(''),
            hex.slice(8, 10).join(''),
            hex.slice(10, 16).join('')
        ].join('-');
    }
    
    // A new ID is generated every time a chat page is opened.
    const tabChatId = createChatId();

    function setStopControl(isActive) {
        if (!stopButton) return;

        stopButton.hidden = !isActive;
        stopButton.disabled = false;
        stopButton.textContent = 'Stop';
    }

    function notifyServerCancellation(requestState) {
        return fetch(chatUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({
                cancel_request: true,
                request_id: requestState.requestId,
                chat_id: requestState.chatId
            }),
            keepalive: true
        }).catch(function(error) {
            console.warn('Could not notify the server about cancellation:', error);
        });
    }

    function abortActiveRequest() {
        if (!activeRequest || activeRequest.stopping) return false;

        activeRequest.stopping = true;
        activeRequest.controller.abort();
        void notifyServerCancellation(activeRequest);
        return true;
    }

    function exportTimestamp() {
        return new Date()
            .toISOString()
            .replace(/[-:]/g, '')
            .replace(/\..+$/, '')
            .replace('T', '_');
    }

    function waitForImages(element) {
        const images = Array.from(element.querySelectorAll('img'));

        return Promise.all(images.map(function(image) {
            if (image.complete) return Promise.resolve();

            return new Promise(function(resolve) {
                image.addEventListener('load', resolve, { once: true });
                image.addEventListener('error', resolve, { once: true });
            });
        }));
    }

    async function clonePlotForPdf(plotElement) {
        const fallbackClone = plotElement.cloneNode(true);

        try {
            const width = Math.max(plotElement.clientWidth, 720);
            const height = Math.max(plotElement.clientHeight, 420);
            const dataUrl = await Plotly.toImage(plotElement, {
                format: 'png',
                width: width,
                height: height,
                scale: 2
            });
            const image = document.createElement('img');
            image.src = dataUrl;
            image.alt = plotElement.getAttribute('aria-label') || 'VasQ plot';
            image.style.display = 'block';
            image.style.width = '100%';
            image.style.height = 'auto';
            image.style.background = '#ffffff';
            return image;
        } catch (error) {
            console.warn('Could not convert Plotly graph to an image:', error);
            return fallbackClone;
        }
    }

    async function downloadMessageRangeAsPdf(startNode, endNode, filename) {
        if (!startNode || !endNode) {
            throw new Error('There are no messages to export.');
        }

        const exportHost = document.createElement('div');
        exportHost.className = 'vasq-pdf-host';

        const exportDocument = document.createElement('div');
        exportDocument.className = 'vasq-pdf-document';
        exportDocument.style.width = Math.max(
            720,
            Math.min(messagesContainer.clientWidth || 900, 1000)
        ) + 'px';
        exportHost.appendChild(exportDocument);

        let currentNode = startNode;
        let reachedEnd = false;

        while (currentNode) {
            if (currentNode.classList.contains('vasq-plot-card')) {
                exportDocument.appendChild(
                    await clonePlotForPdf(currentNode)
                );
            } else {
                exportDocument.appendChild(currentNode.cloneNode(true));
            }

            if (currentNode === endNode) {
                reachedEnd = true;
                break;
            }
            currentNode = currentNode.nextElementSibling;
        }

        if (!reachedEnd) {
            throw new Error('Could not identify the messages for this export.');
        }

        document.body.appendChild(exportHost);

        try {
            if (document.fonts && document.fonts.ready) {
                await document.fonts.ready;
            }
            await waitForImages(exportDocument);

            // Give the off-screen document two paint frames before html2canvas
            // clones it. The document itself remains in normal flow so the
            // html2pdf clone receives its full width and height.
            await new Promise(function(resolve) {
                requestAnimationFrame(function() {
                    requestAnimationFrame(resolve);
                });
            });

            if (!exportDocument.offsetWidth || !exportDocument.offsetHeight) {
                throw new Error('The PDF document did not render.');
            }

            const longestSide = Math.max(
                exportDocument.scrollWidth,
                exportDocument.scrollHeight
            );
            const canvasScale = Math.min(2, 28000 / longestSide);

            const opt = {
                margin: [10, 10, 10, 10],
                filename: filename,
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: {
                    scale: Math.max(0.5, canvasScale),
                    useCORS: true,
                    backgroundColor: '#ffffff',
                    logging: false,
                    windowWidth: exportDocument.scrollWidth,
                    windowHeight: exportDocument.scrollHeight
                },
                jsPDF: {
                    unit: 'mm',
                    format: 'a4',
                    orientation: 'portrait'
                },
                pagebreak: {
                    mode: ['css', 'legacy'],
                    avoid: ['.user-message', '.queue-status-message', '.vasq-plot-card']
                }
            };

            await html2pdf().set(opt).from(exportDocument).save();
        } finally {
            exportHost.remove();
        }

        return filename;
    }

    // Export all currently displayed messages as a PDF.
    document.getElementById('export-pdf').addEventListener('click', async function() {
        const firstMessage = messagesContainer.firstElementChild;
        const lastMessage = messagesContainer.lastElementChild;

        if (!firstMessage || !lastMessage) return;

        try {
            await downloadMessageRangeAsPdf(
                firstMessage,
                lastMessage,
                'chat_history.pdf'
            );
        } catch (error) {
            console.error('Could not export chat PDF:', error);
            addQueueStatus('The PDF could not be generated. Please try again.');
        }
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

        function splitMarkdownTableRow(line) {
            let value = line.trim();

            if (value.startsWith('|')) value = value.slice(1);
            if (value.endsWith('|')) value = value.slice(0, -1);

            return value.split('|').map(function(cell) {
                return cell.trim().replace(/\\_/g, '_');
            });
        }

        function readLine(startIndex) {
            const lineEnd = text.indexOf('\n', startIndex);
            const end = lineEnd === -1 ? text.length : lineEnd;

            return {
                value: text.slice(startIndex, end),
                nextIndex: lineEnd === -1 ? text.length : lineEnd + 1
            };
        }

        function parseMarkdownTable(startIndex) {
            const headerLine = readLine(startIndex);
            if (!headerLine.value.includes('|')) return null;
            if (headerLine.nextIndex >= text.length) return null;

            const dividerLine = readLine(headerLine.nextIndex);
            const headers = splitMarkdownTableRow(headerLine.value);
            const dividers = splitMarkdownTableRow(dividerLine.value);
            const dividerPattern = /^:?-{3,}:?$/;

            if (
                headers.length < 2 ||
                dividers.length !== headers.length ||
                !dividers.every(function(value) {
                    return dividerPattern.test(value);
                })
            ) {
                return null;
            }

            const alignments = dividers.map(function(value) {
                if (value.startsWith(':') && value.endsWith(':')) {
                    return 'center';
                }
                if (value.endsWith(':')) return 'right';
                return 'left';
            });
            const table = document.createElement('table');
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');

            headers.forEach(function(value, columnIndex) {
                const cell = document.createElement('th');
                cell.textContent = value;
                cell.style.textAlign = alignments[columnIndex];
                headerRow.appendChild(cell);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);

            const tbody = document.createElement('tbody');
            let cursor = dividerLine.nextIndex;

            while (cursor < text.length) {
                const dataLine = readLine(cursor);
                const trimmedLine = dataLine.value.trim();

                if (!trimmedLine || !trimmedLine.includes('|')) break;

                const values = splitMarkdownTableRow(dataLine.value);
                if (values.length !== headers.length) break;

                const row = document.createElement('tr');
                values.forEach(function(value, columnIndex) {
                    const cell = document.createElement('td');
                    cell.textContent = value;
                    cell.style.textAlign = alignments[columnIndex];
                    row.appendChild(cell);
                });
                tbody.appendChild(row);
                cursor = dataLine.nextIndex;
            }

            table.appendChild(tbody);
            return { element: table, nextIndex: cursor };
        }

        while (i < text.length) {
            const atLineStart =
                i === 0 || text.charAt(i - 1) === '\n';

            if (atLineStart) {
                const parsedTable = parseMarkdownTable(i);
                if (parsedTable) {
                    fragment.appendChild(parsedTable.element);
                    i = parsedTable.nextIndex;
                    continue;
                }
            }
    
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
        const lines = rawText.split(/\r?\n/);
        const itemPattern = /^\s*(?:\d+[.)]|[-*\u2022])\s+(.+)$/;
        const hasMarkedItems = lines.some(function(line) {
            return itemPattern.test(line);
        });

        // Without numbering/bullets, every non-empty line is one question.
        if (!hasMarkedItems) {
            return lines.map(function(line) {
                return line.trim();
            }).filter(Boolean);
        }

        // A numbered/bulleted question may continue onto following lines.
        const questions = [];
        let currentQuestion = '';

        lines.forEach(function(line) {
            const match = line.match(itemPattern);

            if (match) {
                if (currentQuestion) questions.push(currentQuestion);
                currentQuestion = match[1].trim();
                return;
            }

            const continuation = line.trim();
            if (continuation && currentQuestion) {
                currentQuestion += ` ${continuation}`;
            }
        });

        if (currentQuestion) questions.push(currentQuestion);
        return questions;
    }

    function secondsSince(startedAt) {
        return Math.round((performance.now() - startedAt) / 100) / 10;
    }

    function downloadBatchResults(results, requestedCount, wasStopped) {
        const successful = results.filter(function(result) {
            return result.success;
        }).length;
        const failed = results.length - successful;
        const lines = [
            'VasQ Batch Test Results',
            'Generated: ' + new Date().toISOString(),
            'Questions requested: ' + requestedCount,
            'Questions processed: ' + results.length,
            'Successful: ' + successful,
            'Failed: ' + failed,
            'Stopped early: ' + (wasStopped ? 'yes' : 'no'),
            ''
        ];

        results.forEach(function(result, index) {
            lines.push('='.repeat(78));
            lines.push(`Question ${index + 1}`);
            lines.push('='.repeat(78));
            lines.push('Question:');
            lines.push(result.question);
            lines.push('');
            lines.push('Status: ' + (result.success ? 'SUCCESS' : 'FAILED'));
            lines.push('Elapsed seconds: ' + result.elapsedSeconds.toFixed(1));
            lines.push('Graph returned: ' + (result.hasGraph ? 'yes' : 'no'));
            lines.push('');
            lines.push('Answer:');
            lines.push(result.answer || '(No answer returned)');
            lines.push('');
        });

        // UTF-8 BOM keeps Chinese text readable in Windows Notepad and Excel.
        const blob = new Blob(
            ['\uFEFF' + lines.join('\n')],
            { type: 'text/plain;charset=utf-8' }
        );
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        const timestamp = exportTimestamp();
        const filename = `vasq_batch_results_${timestamp}.txt`;

        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        return filename;
    }

    function setQueueControls(isRunning) {
        queueRunning = isRunning;
        messageInput.disabled = isRunning;
        sendButton.disabled = isRunning;

        if (isRunning) {
            queueButton.textContent = 'Stop Queue';
            queueButton.disabled = false;
        } else {
            queueButton.textContent = 'Run Queue';
            queueButton.disabled = false;
        }
    }

    async function renderGraph(graphJsonValue) {
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
            await Plotly.newPlot(
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
        const startedAt = performance.now();
        const queueIndex = options.queueIndex || null;
        const queueTotal = options.queueTotal || null;
        const resetHistory = options.resetHistory === true;
        const isQueueRequest = Boolean(queueIndex && queueTotal);
        const requestId = createChatId();
        const requestChatId = options.chatId || tabChatId;
        const controller = new AbortController();
        const requestState = {
            controller: controller,
            requestId: requestId,
            chatId: requestChatId,
            stopping: false
        };
        const prefix = queueIndex && queueTotal
            ? `[${queueIndex}/${queueTotal}] `
            : '';

        activeRequest = requestState;
        if (!isQueueRequest) setStopControl(true);

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
                    reset_history: resetHistory,
                    request_id: requestId,
                    chat_id: requestChatId
                }),
                signal: controller.signal
            });

            const data = await response.json().catch(function() {
                return {};
            });

            if (data.stopped) {
                const stoppedError = new Error('Stopped by user.');
                stoppedError.name = 'AbortError';
                throw stoppedError;
            }

            if (!response.ok) {
                throw new Error(
                    data.response || `Server returned HTTP ${response.status}`
                );
            }

            const answerText = data.response || 'I could not generate a response.';
            addChatMessage(answerText, false);

            await renderGraph(data.graph_json);
            return {
                success: true,
                stopped: false,
                answer: answerText,
                hasGraph: Boolean(data.graph_json),
                elapsedSeconds: secondsSince(startedAt)
            };
        } catch (error) {
            if (error.name === 'AbortError') {
                addChatMessage(`${prefix}Stopped.`, false);
                return {
                    success: false,
                    stopped: true,
                    answer: 'Stopped by user.',
                    hasGraph: false,
                    elapsedSeconds: secondsSince(startedAt)
                };
            }

            console.error('Error:', error);
            const errorText = error.message || 'Could not send this question.';
            addChatMessage(
                `${prefix}Failed: ${errorText}`,
                false
            );
            return {
                success: false,
                stopped: false,
                answer: errorText,
                hasGraph: false,
                elapsedSeconds: secondsSince(startedAt)
            };
        } finally {
            if (activeRequest === requestState) {
                activeRequest = null;
            }
            if (!isQueueRequest) setStopControl(false);
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
        stopButton &&
        queueButton
    ) {
        stopButton.addEventListener('click', function() {
            if (!abortActiveRequest()) return;

            stopButton.textContent = 'Stopping...';
            stopButton.disabled = true;
        });

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
                abortActiveRequest();
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
            const queueStartMessage = addQueueStatus(
                `Queue started: ${questions.length} question(s).`
            );

            let completed = 0;
            let failed = 0;
            const results = [];

            for (let index = 0; index < questions.length; index += 1) {
                if (stopQueueRequested) break;

                const result = await sendQuestion(questions[index], {
                    queueIndex: index + 1,
                    queueTotal: questions.length,
                    resetHistory: true,
                    chatId: createChatId()
                });

                if (result.stopped) {
                    stopQueueRequested = true;
                    break;
                }

                results.push({
                    question: questions[index],
                    success: result.success,
                    answer: result.answer,
                    hasGraph: result.hasGraph,
                    elapsedSeconds: result.elapsedSeconds
                });

                if (result.success) {
                    completed += 1;
                } else {
                    failed += 1;
                }
            }

            const wasStopped = stopQueueRequested;
            const pdfFilename =
                `vasq_batch_results_${exportTimestamp()}.pdf`;
            const completionText = wasStopped
                ? `Queue stopped. Completed: ${completed}; failed: ${failed}. `
                : `Queue finished. Completed: ${completed}; failed: ${failed}. `;
            const queueEndMessage = addQueueStatus(
                completionText + `PDF: ${pdfFilename}`
            );

            try {
                await downloadMessageRangeAsPdf(
                    queueStartMessage,
                    queueEndMessage,
                    pdfFilename
                );
            } catch (error) {
                console.error('Could not export queue PDF:', error);
                const fallbackFilename = downloadBatchResults(
                    results,
                    questions.length,
                    wasStopped
                );
                queueEndMessage.textContent =
                    completionText +
                    `PDF export failed; text fallback downloaded: ${fallbackFilename}`;
            }

            stopQueueRequested = false;
            setQueueControls(false);
            messageInput.focus();
        });
    }
    
});
