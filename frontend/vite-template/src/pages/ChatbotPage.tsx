import { useEffect, useState } from "react";
import { Button } from "@heroui/button";
import { PlusFilledIcon, SearchIcon } from "@heroui/shared-icons";
import { v4 as uuidv4 } from "uuid";

import { isMock, mockChat, mockChats } from "../mocks";
import LeftCard from "../components/LeftCard";
import InputCustom from "../components/InputCustom";
import { BACKEND_URL } from "../consts";
import getAlert from "../helpers/getAlert.jsx";
import Message from "../components/Message";

type SourceDocument = {
  content: string;
  source: string;
  file_name: string;
  chunk_id: number;
  page: number;
  is_semantic_chunk: boolean;
};

type ResponseData = {
  think: string;
  theme: string;
  answer: string;
};

type Response = {
  human_handoff: boolean;
  conversation_id: string;
  source_documents: SourceDocument[];
  used_files: string[];
  response: ResponseData;
};

type generateMessageType = {
  user_id: string;
  chat_id: number;
  query_id: string;
  query: string;
  response: Response;
  category: string;
  time: number;
  used_files: string[];
};

const ChatbotPage = () => {
  const user_id = "1";
  const chat_id = 1;
  // список всех чатов пользователя
  const [chats, setChats] = useState<any[]>(isMock(mockChats, []));
  const [query, setQuery] = useState<string>("");
  // объект с смс-ами польвателя в текущем чате
  const [chat, setChat] = useState<object[]>(isMock(mockChat, []));
  // spin у сообщения при отправке
  const [isFetching, setIsFetching] = useState<{isFetching: boolean, id: string}>({
    isFetching: false,
    id: "",
  });

  const handleSend = (e) => {
    const mes_id = uuidv4();
    console.log('mes_id:', mes_id);

    console.log("click", query);
    if (!query) {
      return null;
    }
    console.log("click2", query);
    setIsFetching({ isFetching: true, id: mes_id });
    setChat((prev) => [...prev, { text: query, type: "user", id: mes_id }]);

    console.log('messages:', chat);

    fetch(`${BACKEND_URL}/api/v1/query/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, user_id: "1", chat_id: 1 }),
      })
        .then((res) => res.json())
        .then((ans: generateMessageType) => {
          const message = {
            text: ans?.response?.response?.answer,
            type: "bot",
            id: ans?.query_id,
            source: ans?.response?.used_files,
          };
          console.log('message:', message)
  
          if (message) {
            setChat((prev) => [...prev, message ? message : []]);
          }
          setIsFetching({ isFetching: false, id: "" });
        })
        .catch((e) => {
          getAlert(e.message);
          setIsFetching({ isFetching: false, id: "" });
        });
        
  };

  useEffect(() => {
    fetch(`${BACKEND_URL}/chat?user_id=${user_id}&chatId=${chat_id}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      // body: JSON.stringify({user_id, chat_id})
    })
      .then((res) => res.json())
      .then((ans: object[]) => {
        setChat((prev) => [...prev, ...(Array.isArray(ans) ? ans : [])]);
      })
      .catch((e) => getAlert(e.message));
  }, []);

  console.log("chatMessage:", chat);
  const chatMessages = chat.map((item, key) => (
    <Message
      key={key}
      id={item?.id}
      isFetching={isFetching}
      position={item?.type === "user" ? "end" : "start"}
      text={item?.text ?? isMock("Замоканное сообщение", "Сообщение пустое")}
      type={item?.type ?? "user"}
      source={item?.source ?? undefined}
    />
  ));
  const chatsCards = chats.map((item) => (
    <LeftCard
      text={item?.text ? item.text : isMock("Как составить деклорацию", "")}
    />
  ));

  return (
    <>
    <div className="flex flex-row justify-stretch hello w-[100%]">
      <div className="flex flex-col rounded-16 w-[360px] bg-grey-background flex-4 mt-[20px] ml=[20px]
        border-0
        border-solid
        border-color: currentColor;
      ">
        {/* <div> */}
        <div className="relative bg-white rounded-cust h-[100vh]">
        <div className="h-[100%] bg-white rounded-cust pt-[80px]">
            <div className="flex flex-row justify-around mb-[20px]">
            <div className="flex-3">
            <Button color="primary"
             style={{ width: '138%' }}
            >
                <PlusFilledIcon />
                Новый чат
            </Button>
          </div>
          <Button isIconOnly aria-label="Like" color="danger">
            <SearchIcon />
          </Button>
          </div>
          <p className="text-left pl-[30px] mb-[10px]">Другие чаты</p>
          <div className="flex flex-col gap-[-3px]">
            {chatsCards}
        </div>
        </div>
        </div>
        {/* </div> */}
      </div>
      <div className="h-[100vh] min-h-full flex flex-8 flex-col w-[100%] bg-grey-background p-[20px] bottom-[10px] overflow-y: auto">
        <h1 className="mb-[50px]">Начните диалог</h1>
        <div className="flex flex-col justify-left gap-[24px]">
          {chatMessages}
          <div className="sticky bottom-[0px] left-[500px] relative w-[900px]">
          <div className="relative">
            <InputCustom icon={SearchIcon} onChange={setQuery} />
            <Button
                color="secondary"
              className="absolute top-[0px] right-[200px] h-[50px]"
              onPress={(e) => {
                handleSend(e);
              }}
            >
              <SearchIcon />
            </Button>
          </div>
        </div>
        </div>
      </div>
    </div>
    {/* <div className="sticky bottom-[10px] left-[500px] relative w-[900px]">
          <div className="relative">
            <InputCustom icon={SearchIcon} onChange={setQuery} />
  
            <Button
                color="secondary"
              className="absolute top-[0px] right-[200px] h-[50px]"
              onPress={(e) => {
                handleSend(e);
              }}
            >
              <SearchIcon />
            </Button>
          </div>
        </div> */}
    </>
  );
};

export default ChatbotPage;
