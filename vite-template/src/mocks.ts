import { message } from "antd";

export const IS_MOCKS = true;


export function isMock(data, data2) {
    if (IS_MOCKS) return data;
    return data2;
}

export const mockChats = [
    {id: 1, text: 'Как составить докларацию'},
    {id: 2, text: 'Как составить докларацию'},
    {id: 3, text: 'Как составить докларацию'},
    {id: 3, text: 'Как составить докларацию'},
]

export const mockChat = [
    {text: 'Вопрос пользователя', type: 'user', id: '1'},
    {text: 'Пример ответа нейросети', type: 'bot', id: '2'},
    // {message: 'Третье сообщение', type: 'user', id: '3'},
    // {message: 'Четвертое сообщение', type: 'bot', id: '4'},
    // {message: 'Пятое сообщение', type: 'user', id: '5'},
]

export const mocksData = [
    {
        x: 1,
        y: 0,
        label: 'Навигация и функциональность портала',
    },
    {
        x: 2,
        y: 4,
        label: 'Технические проблемы',
    },
    {
        x: 3,
        y: 3,
        label: 'Документы и инструкции',
    },
    {
        x: 4,
        y: 2,
        label: 'Законодательство и нормативка',
    },
    {
        x: 5,
        y: 2,
        label: 'Обратная связь и жалобы',
    },
];

export const mockVictoryPie = [
    { x: "Навигация и функциональность портала", y: 35 },
    { x: "Технические проблемы", y: 40 },
    { x: "Документы и инструкции", y: 55 },
    { x: "Законодательство и нормативка", y: 35 },
    { x: "Обратная связь и жалобы", y: 35 }
];