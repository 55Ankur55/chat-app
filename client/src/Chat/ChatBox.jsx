import React, { useState, useEffect, useRef } from "react";
import { makeStyles } from "@material-ui/core/styles";
import Grid from "@material-ui/core/Grid";
import Typography from "@material-ui/core/Typography";
import TextField from "@material-ui/core/TextField";
import IconButton from "@material-ui/core/IconButton";
import SendIcon from "@material-ui/icons/Send";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import ListItemAvatar from "@material-ui/core/ListItemAvatar";
import Avatar from "@material-ui/core/Avatar";
import Paper from "@material-ui/core/Paper";
import socketIOClient from "socket.io-client";
import classnames from "classnames";
import commonUtilites from "../Utilities/common";
import {
   useGetGlobalMessages,
   useSendGlobalMessage,
   useGetConversationMessages,
   useSendConversationMessage,
} from "../Services/chatService";
import { authenticationService } from
"../Services/authenticationService";

const useStyles = makeStyles((theme) => ({
   root: {
     height: "100%",
   },
   headerRow: {
     maxHeight: 60,
     zIndex: 5,
   },
   paper: {
     display: "flex",
     alignItems: "center",
     justifyContent: "center",
     height: "100%",
     color: theme.palette.primary.dark,
   },
   messageContainer: {
     height: "100%",
     display: "flex",
     alignContent: "flex-end",
   },
   messagesRow: {
     maxHeight: "calc(100vh - 184px)",
     overflowY: "auto",
   },
   newMessageRow: {
     width: "100%",
     padding: theme.spacing(0, 2, 1),
   },
   messageBubble: {
     padding: 10,
     border: "1px solid white",
     backgroundColor: "white",
     borderRadius: "0 10px 10px 10px",
     boxShadow: "-3px 4px 4px 0px rgba(0,0,0,0.08)",
     marginTop: 8,
     maxWidth: "40em",
   },
   messageBubbleRight: {
     borderRadius: "10px 0 10px 10px",
   },
   inputRow: {
     display: "flex",
     alignItems: "flex-end",
   },
   form: {
     width: "100%",
   },
   avatar: {
     margin: theme.spacing(1, 1.5),
   },
   listItem: {
     display: "flex",
     width: "100%",
   },
   listItemRight: {
     flexDirection: "row-reverse",
   },
}));

const ChatBox = (props) => {
     /* alert(props.setScope) */

   const [currentUserId] = useState(
     authenticationService.currentUserValue.userId
   );
   const [newMessage, setNewMessage] = useState("");
   const [messages, setMessages] = useState([]);
   const [lastMessage, setLastMessage] = useState(null);

   //sushil
   const [botmessages, setNewMessagesss] = useState("");

   const getGlobalMessages = useGetGlobalMessages();
   const sendGlobalMessage = useSendGlobalMessage();
   const getConversationMessages = useGetConversationMessages();
   const sendConversationMessage = useSendConversationMessage();
   let message = '';
   let chatBottom = useRef(null);
   const classes = useStyles();

   useEffect(() => {
     reloadMessages();
     scrollToBottom();
   }, [lastMessage, props.scope, props.conversationId]);


   /* useEffect(() => {

     const socket = socketIOClient(process.env.REACT_APP_API_URL);
     socket.on("messages", (data) => setLastMessage(data));
     socket.on("botmessages", (data) => console.log(data));

   }, []); */

   useEffect(() => {
     const socket = socketIOClient(process.env.REACT_APP_API_URL);
     socket.on("messages", (data) => setLastMessage(data));
     socket.on("botmessages", (data) =>

       setNewMessagesss(data));





   }, []);

   const reloadMessages = () => {
     if (props.scope === "Admin") {
       // getGlobalMessages().then((res) => {
       //   setMessages(res);
       // });
     } else if (props.scope !== null && props.conversationId !== null) {
       getConversationMessages(props.user._id).then((res) =>
setMessages(res));
     } else {
       setMessages([]);
     }
   };

   const scrollToBottom = () => {
     chatBottom.current.scrollIntoView({ behavior: "smooth" });
   };

   useEffect(scrollToBottom, [messages]);

   const handleSubmit = (e) => {
     // alert(botmessagessssss)
     e.preventDefault();
     if(newMessage!=""){
       if (props.scope === "Global Chat") {
         sendGlobalMessage(newMessage).then(() => {
           setNewMessage("");
         });
       } else {
         sendConversationMessage(props.user._id, newMessage).then((res)=> {
           setNewMessage("");
         });
       }
     }
     setNewMessagesss("");
   };

   return (
     <Grid container className={classes.root}>
       <Grid item xs={12} className={classes.headerRow}>
         <Paper className={classes.paper} square elevation={2}>
           <Typography color="inherit" variant="h6">
             {props.scope}
           </Typography>
         </Paper>
       </Grid>
       <Grid item xs={12}>
         <Grid container className={classes.messageContainer}>
           <Grid item xs={12} className={classes.messagesRow}>
             {messages && (
               <List>
                 {messages.map((m) => (
                   <ListItem
                     key={m._id}
                     className={classnames(classes.listItem, {
                       [`${classes.listItemRight}`]:
                         m.fromObj[0]._id === currentUserId,
                       })}
                       alignItems="flex-start"
                   >
                     {/* <ListItem
                     key={m._id}
                     className={classnames(classes.listItem, {
                       [`${classes.listItemRight}`]:
                         m.fromObj[0]._id === currentUserId,
                     })}
                     {...botmessages}
                     alignItems="flex-start"
                   ></ListItem> */}
                     <ListItemAvatar className={classes.avatar}>
                       <Avatar>

{commonUtilites.getInitialsFromName(m.fromObj[0].name)}
                       </Avatar>
                     </ListItemAvatar>
                     <ListItemText
                       classes={{
                         root: classnames(classes.messageBubble, {
                           [`${classes.messageBubbleRight}`]:
                             m.fromObj[0]._id === currentUserId,
                         }),
                       }}
                       primary={m.fromObj[0] && m.fromObj[0].name}

secondary={<React.Fragment>{m.body}</React.Fragment>}
                     />

                   </ListItem>
                 ))}
                   { botmessages &&
                   <>
                     <ListItem
                    /*  key={m._id} */
                     className={classnames(classes.listItem, {
                     //  [`${classes.listItemRight}`]:"BOT",
                     })}
                     alignItems="flex-start"
                   >
                 <ListItemAvatar className={classes.avatar}>
                   <Avatar>
                     {"B"}
                   </Avatar>
                 </ListItemAvatar>
                 <ListItemText

                   /* className={classes.messageBubble}

secondary={<React.Fragment>{botmessages}</React.Fragment>} */

                   classes={{
                     root: classnames(classes.messageBubble, {
                        // [`${classes.messageBubbleRight}`]:
                         // m.fromObj[0]._id === currentUserId,
                     }),
                   }}
                    primary={"BOT"} 

secondary={<React.Fragment>{botmessages.replace(/'/g,'').replace(/[[\]]/g,'')}</React.Fragment>}
                 />
                 </ListItem>
                 </>                }

               </List>
             )}
             <div ref={chatBottom} />
           </Grid>
           <Grid item xs={12} className={classes.inputRow}>
             <form onSubmit={handleSubmit} className={classes.form}>
               <Grid
                 container
                 className={classes.newMessageRow}
                 alignItems="flex-end"
               >
                 <Grid item xs={11}>
                   <TextField
                     id="message"
                     label="Message"
                     variant="outlined"
                     margin="dense"
                     fullWidth
                     value={newMessage}
                     onChange={(e) => setNewMessage(e.target.value)}
                   />
                 </Grid>
                 <Grid item xs={1}>
                   <IconButton type="submit">
                     <SendIcon />
                   </IconButton>
                 </Grid>
               </Grid>
             </form>
           </Grid>
         </Grid>
       </Grid>
     </Grid>
   );
};

export default ChatBox;

