import React, {Component} from "react";
import {Button, Icon, Label, Popup} from "semantic-ui-react";
import {BASE_URL, getAuthorization, refreshJWT} from "../Util";

export default class AgentLikes extends Component {
  // props:
  // agent: Agent
  // me: User
  constructor(props) {
    super(props);
    this.state = {
      likes: 0,
      liked: false,
      like_id: -1,
    };
  }

  componentDidMount() {
    this.loadData();
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (prevProps.me?.id !== this.props.me?.id) {
      this.loadData();
    }
  }

  loadData() {
    if (this.props.agent === undefined) return;

    fetch(BASE_URL + '/api/agent-likes/?count&agent_id=' + this.props.agent.id, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...getAuthorization(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        console.error("Unable to fetch agent likes: ", response);
        return;
      }

      response.json().then(likes => {
        this.setState({likes: likes.count});
      })
    });

    if (this.props.me !== undefined) {
      fetch(BASE_URL + '/api/agent-likes/?liked&agent_id=' + this.props.agent.id, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          ...getAuthorization(),
        },
      }).then(response => {
        if (response.status === 401) {
          refreshJWT();
          return;
        }

        if (response.status !== 200) {
          console.error("Unable to fetch whether you liked the agent: ", response);
          return;
        }

        response.json().then(liked => {
          this.setState({
            liked: liked.liked,
            like_id: liked.like_id,
          });
        });
      });
    }
  }

  createLike = () => {
    if (this.props.me === undefined) return;
    fetch(BASE_URL + '/api/agent-likes/', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...getAuthorization(),
      },
      body: JSON.stringify({
        agent_id: this.props.agent.id,
        user_id: this.props.me.id,
      }),
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 201) {
        console.error("Unable to like the agent: ", response);
        return;
      }

      response.json().then(like => {
        this.setState({
          liked: true,
          likes: this.state.likes + 1,
          like_id: like.id,
        });
      });
    });
  };

  removeLike = () => {
    if (this.props.me === undefined) return;
    fetch(BASE_URL + '/api/agent-likes/' + this.state.like_id + '/', {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...getAuthorization(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 204) {
        console.error("Unable to unlike the agent: ", response);
        return;
      }

      this.setState({
        liked: false,
        likes: this.state.likes - 1,
      });
    });
  };

  render() {
    if (this.props.me === undefined) {
      return (
        <Popup
          trigger={this.likeButton()}
          position='bottom center'
          flowing
          hoverable
        >
          You must <a href='/signin'>sign in</a> to like
        </Popup>
      );
    }

    return this.likeButton();
  }

  likeButton = () => (
    <Button as='div' labelPosition='right' size='mini' onClick={() => {
      if (this.state.liked) {
        this.removeLike();
      } else {
        this.createLike();
      }
    }}>
      <Button color='teal' size='mini'>
        <Icon name='thumbs up'/>
      </Button>
      <Label as='a' basic pointing='left'>
        {this.state.likes}
      </Label>
    </Button>
  );
}
